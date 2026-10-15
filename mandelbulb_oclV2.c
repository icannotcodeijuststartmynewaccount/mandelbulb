#include "mandelbulb_oclV2.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef USE_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

static const char *KERNEL_SRC =
"#ifdef USE_DOUBLE\n"
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
"typedef double real;\n"
"typedef double3 real3;\n"
"#else\n"
"typedef float real;\n"
"typedef float3 real3;\n"
"#endif\n"
"#define LIT(x) ((real)(x))\n"
"\n"
"static real de_fn(real3 c, int n, int maxIter, real bail2, real deSafety)\n"
"{\n"
"    const real tiny = LIT(1e-30);\n"
"    real3 z = c;\n"
"    real dr = LIT(1.0);\n"
"    for (int i = 0; i < maxIter; i++) {\n"
"        real r2 = dot(z,z);\n"
"        if (r2 > bail2) break;\n"
"        real r = sqrt(r2), rs = fmax(r, tiny);\n"
"        real rxy = sqrt(z.x*z.x + z.y*z.y), rxs = fmax(rxy, tiny);\n"
"        real ct = clamp(z.z/rs, LIT(-1.0), LIT(1.0));\n"
"        real st = rxy/rs, cp = z.x/rxs, sp = z.y/rxs;\n"
"        real two_ct = LIT(2.0)*ct;\n"
"        real Tp=LIT(1.0),Tc=ct,Up=LIT(1.0),Uc=two_ct;\n"
"        for (int k=2;k<=n;k++){real Tn=two_ct*Tc-Tp;real Un=two_ct*Uc-Up;Tp=Tc;Tc=Tn;Up=Uc;Uc=Un;}\n"
"        real cnt=Tc, snt=st*Up;\n"
"        real two_cp = LIT(2.0)*cp;\n"
"        Tp=LIT(1.0);Tc=cp;Up=LIT(1.0);Uc=two_cp;\n"
"        for (int k=2;k<=n;k++){real Tn=two_cp*Tc-Tp;real Un=two_cp*Uc-Up;Tp=Tc;Tc=Tn;Up=Uc;Uc=Un;}\n"
"        real cnp=Tc, snp=sp*Up;\n"
"        real rn=r; for(int k=1;k<n;k++) rn*=r;\n"
"        real rn1=rn/rs;\n"
"        dr = (real)n*rn1*dr + LIT(1.0);\n"
"        real stn = rn*snt;\n"
"        z.x = stn*cnp + c.x;\n"
"        z.y = stn*snp + c.y;\n"
"        z.z = rn *cnt + c.z;\n"
"    }\n"
"    real rf = fmax(sqrt(dot(z,z)), tiny);\n"
"    return deSafety * LIT(0.5) * log(rf) * rf / dr;\n"
"}\n"
"\n"
"static real3 calc_normal(real3 p, int n, int maxIter, real bail2, real deSafety)\n"
"{\n"
"    const real h = LIT(0.0005)*LIT(0.5773502691896258);\n"
"    real3 o1=(real3)(h,-h,-h), o2=(real3)(-h,-h,h), o3=(real3)(-h,h,-h), o4=(real3)(h,h,h);\n"
"    real3 nn = o1*de_fn(p+o1,n,maxIter,bail2,deSafety)\n"
"             + o2*de_fn(p+o2,n,maxIter,bail2,deSafety)\n"
"             + o3*de_fn(p+o3,n,maxIter,bail2,deSafety)\n"
"             + o4*de_fn(p+o4,n,maxIter,bail2,deSafety);\n"
"    return normalize(nn);\n"
"}\n"
"\n"
"static real calc_ao(real3 p, real3 nn, int n, int maxIter, real bail2, real deSafety)\n"
"{\n"
"    real occ = LIT(0.0);\n"
"    for (int i = 0; i < 8; i++) {\n"
"        real fi = (real)i;\n"
"        real h = LIT(0.01) + LIT(0.15) * fi * fi / LIT(49.0);\n"
"        real d = de_fn(p + nn * h, n, maxIter, bail2, deSafety);\n"
"        occ += (h - d) * (LIT(1.0) - fi / LIT(8.0));\n"
"    }\n"
"    return clamp(LIT(1.0) - LIT(2.5)*occ, LIT(0.0), LIT(1.0));\n"
"}\n"
"\n"
"static real trace_shadow(real3 p, real3 lig, int n, int maxIter, real bail2, real deSafety)\n"
"{\n"
"    real s = LIT(1.0);\n"
"    real t = LIT(0.02);\n"
"    for (int i = 0; i < 24; i++) {\n"
"        real3 q = p + lig*t;\n"
"        real d = de_fn(q, n, maxIter, bail2, deSafety);\n"
"        s = fmin(s, LIT(8.0)*d/t);\n"
"        t += fmax(d, LIT(0.02));\n"
"        if (s < LIT(0.01)) break;\n"
"        if (d < LIT(1e-7)) return LIT(0.0);\n"
"        if (t > LIT(4.0)) break;\n"
"    }\n"
"    return clamp(s, LIT(0.0), LIT(1.0));\n"
"}\n"
"\n"
"__kernel void mandelbulb_kernel(\n"
"    __global uchar *out,\n"
"    const int width, const int height,\n"
"    const real camX, const real camY, const real camZ,\n"
"    const real crx, const real cry, const real crz,\n"
"    const real cux, const real cuy, const real cuz,\n"
"    const real cfx, const real cfy, const real cfz,\n"
"    const real aspect,\n"
"    const int  n, const int maxIter, const int maxSteps,\n"
"    const real bail2, const real epsScl, const real epsMin,\n"
"    const real deSafety,\n"
"    const int  newtonIters, const int softShadow,\n"
"    const int  palette,\n"
"    const real dof,\n"
"    const real lightX, const real lightY, const real lightZ)\n"
"{\n"
"    int x = get_global_id(0), y = get_global_id(1);\n"
"    if (x >= width || y >= height) return;\n"
"\n"
"    real fx = ((real)x + LIT(0.5)) / (real)width;\n"
"    real fy = ((real)y + LIT(0.5)) / (real)height;\n"
"    real u  = (LIT(2.0)*fx - LIT(1.0)) * aspect;\n"
"    real v  = LIT(1.0) - LIT(2.0)*fy;\n"
"\n"
"    /* DoF jitter per sub-pixel */\n"
"    real3 jitter = (real3)(LIT(0.0));\n"
"    if (dof > LIT(0.0)) {\n"
"        uint h = (uint)x * 73856093u ^ (uint)y * 19349663u;\n"
"        h = h * 1664525u + 1013904223u;\n"
"        real r1 = (real)((h >> 8) & 0xFFFFu) / LIT(65535.0);\n"
"        h = h * 1664525u + 1013904223u;\n"
"        real r2 = (real)((h >> 8) & 0xFFFFu) / LIT(65535.0);\n"
"        real rr = sqrt(r1) * dof;\n"
"        real th = r2 * LIT(6.283185307179586);\n"
"        jitter = (real3)(rr*cos(th), rr*sin(th), LIT(0.0));\n"
"    }\n"
"    real3 right = normalize((real3)(crx, cry, crz));\n"
"    real3 upv   = normalize((real3)(cux, cuy, cuz));\n"
"    real3 ro = (real3)(camX, camY, camZ) + right*jitter.x + upv*jitter.y;\n"
"    real3 rd = normalize((real3)(cfx + crx*u + cux*v,\n"
"                                 cfy + cry*u + cuy*v,\n"
"                                 cfz + crz*u + cuz*v));\n"
"\n"
"    /* raymarch */\n"
"    real t = LIT(0.0);\n"
"    real hit = LIT(0.0);\n"
"    for (int i = 0; i < maxSteps; i++) {\n"
"        real3 p = ro + rd*t;\n"
"        real d = de_fn(p, n, maxIter, bail2, deSafety);\n"
"        real eps = fmax(epsScl*t, epsMin);\n"
"        if (d < eps) { hit = LIT(1.0); break; }\n"
"        t += d;\n"
"        if (t > LIT(20.0)) break;\n"
"    }\n"
"    /* newton refinement */\n"
"    if (hit > LIT(0.5) && newtonIters > 0) {\n"
"        for (int k = 0; k < newtonIters; k++) {\n"
"            real3 p = ro + rd*t;\n"
"            real dd = de_fn(p, n, maxIter, bail2, deSafety);\n"
"            real3 nn = calc_normal(p, n, maxIter, bail2, deSafety);\n"
"            real denom = dot(nn, rd);\n"
"            if (fabs(denom) < LIT(1e-8)) break;\n"
"            t -= dd / denom;\n"
"        }\n"
"    }\n"
"\n"
"    real3 col;\n"
"    if (hit < LIT(0.5)) {\n"
"        real k = clamp(rd.y*LIT(0.5) + LIT(0.5), LIT(0.0), LIT(1.0));\n"
"        col = (real3)(LIT(0.015) + LIT(0.205)*k,\n"
"                      LIT(0.020) + LIT(0.360)*k,\n"
"                      LIT(0.045) + LIT(0.635)*k);\n"
"    } else {\n"
"        real3 p = ro + rd*t;\n"
"        real3 nn = calc_normal(p, n, maxIter, bail2, deSafety);\n"
"        real3 lig = normalize((real3)(lightX, lightY, lightZ));\n"
"        real dif = fmax(dot(nn, lig), LIT(0.0));\n"
"        if (softShadow) dif *= trace_shadow(p, lig, n, maxIter, bail2, deSafety);\n"
"        real amb = clamp(LIT(0.5) + LIT(0.5)*nn.y, LIT(0.0), LIT(1.0));\n"
"        real ao  = calc_ao(p, nn, n, maxIter, bail2, deSafety);\n"
"\n"
"        real3 base = (real3)(LIT(0.92), LIT(0.58), LIT(0.26));\n"
"        if (palette == 1) {\n"
"            real k = clamp(t*LIT(0.25), LIT(0.0), LIT(1.0));\n"
"            base = (real3)(LIT(0.9) - LIT(0.5)*k, LIT(0.3) + LIT(0.5)*k, LIT(0.1) + LIT(0.7)*k);\n"
"        } else if (palette == 2) {\n"
"            base = (real3)(LIT(0.5) + LIT(0.5)*nn.x,\n"
"                           LIT(0.5) + LIT(0.5)*nn.y,\n"
"                           LIT(0.5) + LIT(0.5)*nn.z);\n"
"        } else if (palette == 3) {\n"
"            base = (real3)(LIT(0.2) + LIT(0.8)*dif,\n"
"                           LIT(0.2) + LIT(0.6)*dif,\n"
"                           LIT(0.2) + LIT(0.4)*dif);\n"
"        }\n"
"        real lam = amb*LIT(0.30)*ao + dif*LIT(0.90);\n"
"        col = base * lam;\n"
"\n"
"        real3 hv = normalize(lig - rd);\n"
"        real spe = pown(fmax(dot(nn, hv), LIT(0.0)), 48);\n"
"        col += (real3)(LIT(1.00),LIT(0.94),LIT(0.82)) * (spe*dif*LIT(0.75));\n"
"        real fre = pown(clamp(LIT(1.0)+dot(nn, rd), LIT(0.0), LIT(1.0)), 4);\n"
"        col += (real3)(LIT(0.28),LIT(0.40),LIT(0.62)) * (fre*LIT(0.35));\n"
"    }\n"
"\n"
"    col.x = col.x/(LIT(1.0)+col.x);\n"
"    col.y = col.y/(LIT(1.0)+col.y);\n"
"    col.z = col.z/(LIT(1.0)+col.z);\n"
"    real g = LIT(1.0)/LIT(2.2);\n"
"    col.x = pow(col.x, g);\n"
"    col.y = pow(col.y, g);\n"
"    col.z = pow(col.z, g);\n"
"    int idx = (y*width + x)*3;\n"
"    out[idx+0] = (uchar)(clamp(col.x,LIT(0.0),LIT(1.0))*LIT(255.0) + LIT(0.5));\n"
"    out[idx+1] = (uchar)(clamp(col.y,LIT(0.0),LIT(1.0))*LIT(255.0) + LIT(0.5));\n"
"    out[idx+2] = (uchar)(clamp(col.z,LIT(0.0),LIT(1.0))*LIT(255.0) + LIT(0.5));\n"
"}\n";

static const char *clerr(cl_int e)
{
    switch (e) {
        case CL_SUCCESS: return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
        case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
        case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
        case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
        case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
        case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
        default: return "CL_<unknown>";
    }
}

static void print_device_info(cl_device_id d, int idx)
{
    char name[256]={0}, vendor[256]={0}, version[256]={0}, drv[256]={0};
    cl_uint cu = 0; cl_ulong mem = 0; cl_device_fp_config fp64cfg = 0;
    clGetDeviceInfo(d, CL_DEVICE_NAME, sizeof(name), name, NULL);
    clGetDeviceInfo(d, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    clGetDeviceInfo(d, CL_DEVICE_VERSION, sizeof(version), version, NULL);
    clGetDeviceInfo(d, CL_DRIVER_VERSION, sizeof(drv), drv, NULL);
    clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, NULL);
    clGetDeviceInfo(d, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);
    clGetDeviceInfo(d, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fp64cfg), &fp64cfg, NULL);
    printf("  [%d] %s\n", idx, name);
    printf("      vendor: %s\n      OpenCL: %s\n      driver: %s\n", vendor, version, drv);
    printf("      compute units: %u\n      global mem: %.1f MB\n", cu, (double)mem/(1024.0*1024.0));
    printf("      fp64: %s\n", fp64cfg ? "yes" : "no");
}

int ocl_list_devices(void)
{
    cl_uint np = 0;
    if (clGetPlatformIDs(0, NULL, &np) != CL_SUCCESS || np == 0) {
        printf("No OpenCL platforms found.\n"); return 0;
    }
    cl_platform_id *plats = (cl_platform_id*)malloc(sizeof(cl_platform_id) * np);
    clGetPlatformIDs(np, plats, NULL);
    int total = 0;
    for (cl_uint p = 0; p < np; p++) {
        char pname[256]={0}, pver[256]={0};
        clGetPlatformInfo(plats[p], CL_PLATFORM_NAME, sizeof(pname), pname, NULL);
        clGetPlatformInfo(plats[p], CL_PLATFORM_VERSION, sizeof(pver), pver, NULL);
        printf("Platform %u: %s  (%s)\n", p, pname, pver);
        cl_device_id devs[32]; cl_uint nd = 0;
        if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 32, devs, &nd) != CL_SUCCESS) continue;
        for (cl_uint i = 0; i < nd; i++) { print_device_info(devs[i], (int)i); total++; }
    }
    free(plats); return total > 0;
}

static int pick_gpu(cl_platform_id *outPlat, cl_device_id *outDev, int require_double)
{
    cl_uint np = 0;
    if (clGetPlatformIDs(0, NULL, &np) != CL_SUCCESS || np == 0) return 0;
    cl_platform_id *plats = (cl_platform_id*)malloc(sizeof(cl_platform_id) * np);
    clGetPlatformIDs(np, plats, NULL);
    for (int pass = 0; pass < 2; pass++) {
        for (cl_uint p = 0; p < np; p++) {
            char pv[256] = {0};
            clGetPlatformInfo(plats[p], CL_PLATFORM_VENDOR, sizeof(pv), pv, NULL);
            if (pass == 0 && !strstr(pv, "NVIDIA") && !strstr(pv, "nvidia")) continue;
            cl_device_id dev = NULL;
            if (clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 1, &dev, NULL) != CL_SUCCESS) continue;
            if (require_double) {
                cl_device_fp_config fpc = 0;
                clGetDeviceInfo(dev, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(fpc), &fpc, NULL);
                if (fpc == 0) continue;
            }
            *outPlat = plats[p]; *outDev = dev; free(plats); return 1;
        }
    }
    free(plats); return 0;
}

int ocl_renderV2(uint8_t *out, int width, int height,
                 double camX, double camY, double camZ,
                 double crx, double cry, double crz,
                 double cux, double cuy, double cuz,
                 double cfx, double cfy, double cfz,
                 int power, int maxIter, int maxSteps,
                 double bailout, double epsScl, double epsMin,
                 double deSafety,
                 int newtonIters, int softShadow,
                 int palette, double dof,
                 double lightX, double lightY, double lightZ,
                 int use_double)
{
    cl_int err;
    cl_platform_id plat; cl_device_id dev;
    if (!pick_gpu(&plat, &dev, use_double)) {
        fprintf(stderr, "No suitable OpenCL GPU found%s\n",
                use_double ? " (double required)" : ""); return 0;
    }
    char devName[256] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(devName), devName, NULL);
    printf("OpenCL device: %s  (%s kernel)\n", devName, use_double ? "double" : "float");

    cl_context ctx = clCreateContext(NULL, 1, &dev, NULL, NULL, &err);
    if (!ctx) { fprintf(stderr, "ctx: %s\n", clerr(err)); return 0; }
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);
    if (!q) { fprintf(stderr, "queue: %s\n", clerr(err)); clReleaseContext(ctx); return 0; }

    const char *srcs[1] = { KERNEL_SRC };
    cl_program prog = clCreateProgramWithSource(ctx, 1, srcs, NULL, &err);
    if (!prog) { fprintf(stderr, "prog: %s\n", clerr(err));
        clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }

    const char *buildOpts = use_double
        ? "-D USE_DOUBLE -cl-fast-relaxed-math -cl-mad-enable"
        : "-cl-fast-relaxed-math -cl-mad-enable";
    err = clBuildProgram(prog, 1, &dev, buildOpts, NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t ll = 0;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, NULL, &ll);
        char *log = (char*)malloc(ll+1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, ll, log, NULL);
        log[ll] = 0;
        fprintf(stderr, "Build failed (%s):\n%s\n", clerr(err), log);
        free(log); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(ctx); return 0;
    }

    cl_kernel k = clCreateKernel(prog, "mandelbulb_kernel", &err);
    if (!k) { fprintf(stderr, "kernel: %s\n", clerr(err));
        clReleaseProgram(prog); clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }

    size_t imgBytes = (size_t)width * (size_t)height * 3u;
    cl_mem buf = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, imgBytes, NULL, &err);
    if (!buf) { fprintf(stderr, "buf: %s\n", clerr(err));
        clReleaseKernel(k); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(ctx); return 0; }

    int arg = 0; cl_int e = 0;
    #define SET(v) do { e |= clSetKernelArg(k, arg++, sizeof(v), &(v)); } while (0)

    SET(buf); SET(width); SET(height);
    double as = (double)width / (double)height;
    double bail2 = bailout * bailout;

    if (use_double) {
        double v;
        #define SD(x) do { v = (double)(x); SET(v); } while (0)
        SD(camX); SD(camY); SD(camZ);
        SD(crx); SD(cry); SD(crz); SD(cux); SD(cuy); SD(cuz); SD(cfx); SD(cfy); SD(cfz);
        SD(as);
        SET(power); SET(maxIter); SET(maxSteps);
        SD(bail2); SD(epsScl); SD(epsMin); SD(deSafety);
        SET(newtonIters); SET(softShadow); SET(palette);
        SD(dof); SD(lightX); SD(lightY); SD(lightZ);
        #undef SD
    } else {
        float v;
        #define SF(x) do { v = (float)(x); SET(v); } while (0)
        SF(camX); SF(camY); SF(camZ);
        SF(crx); SF(cry); SF(crz); SF(cux); SF(cuy); SF(cuz); SF(cfx); SF(cfy); SF(cfz);
        SF(as);
        SET(power); SET(maxIter); SET(maxSteps);
        SF(bail2); SF(epsScl); SF(epsMin); SF(deSafety);
        SET(newtonIters); SET(softShadow); SET(palette);
        SF(dof); SF(lightX); SF(lightY); SF(lightZ);
        #undef SF
    }
    #undef SET
    if (e != CL_SUCCESS) {
        fprintf(stderr, "setarg: %s\n", clerr(e));
        clReleaseMemObject(buf); clReleaseKernel(k); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(ctx); return 0;
    }

    size_t gx = (size_t)((width  + 15)/16)*16;
    size_t gy = (size_t)((height + 15)/16)*16;
    size_t global[2] = { gx, gy };
    size_t local[2]  = { 16, 16 };

    err = clEnqueueNDRangeKernel(q, k, 2, NULL, global, local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "enqueue: %s\n", clerr(err));
        clReleaseMemObject(buf); clReleaseKernel(k); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(ctx); return 0;
    }
    clFinish(q);
    err = clEnqueueReadBuffer(q, buf, CL_TRUE, 0, imgBytes, out, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "readback: %s\n", clerr(err));
        clReleaseMemObject(buf); clReleaseKernel(k); clReleaseProgram(prog);
        clReleaseCommandQueue(q); clReleaseContext(ctx); return 0;
    }
    clReleaseMemObject(buf); clReleaseKernel(k); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    return 1;
}

#else

int ocl_renderV2(uint8_t *out, int width, int height,
                 double camX, double camY, double camZ,
                 double crx, double cry, double crz,
                 double cux, double cuy, double cuz,
                 double cfx, double cfy, double cfz,
                 int power, int maxIter, int maxSteps,
                 double bailout, double epsScl, double epsMin,
                 double deSafety,
                 int newtonIters, int softShadow,
                 int palette, double dof,
                 double lightX, double lightY, double lightZ,
                 int use_double)
{
    (void)out;(void)width;(void)height;
    (void)camX;(void)camY;(void)camZ;
    (void)crx;(void)cry;(void)crz;(void)cux;(void)cuy;(void)cuz;(void)cfx;(void)cfy;(void)cfz;
    (void)power;(void)maxIter;(void)maxSteps;(void)bailout;(void)epsScl;(void)epsMin;(void)deSafety;
    (void)newtonIters;(void)softShadow;(void)palette;(void)dof;(void)lightX;(void)lightY;(void)lightZ;(void)use_double;
    fprintf(stderr, "No OpenCL. Rebuild with -DUSE_OPENCL -lOpenCL.\n");
    return 0;
}

int ocl_list_devices(void){ fprintf(stderr, "No OpenCL support.\n"); return 0; }

#endif