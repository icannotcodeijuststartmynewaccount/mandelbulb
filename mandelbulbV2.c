/* ============================================================================
 *  mandelbulbV2.c
 *
 *  Compile (MinGW-w64 GCC / MSYS2):
 *      gcc -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -ffast-math \
 *          -DUSE_OPENCL -o mandelbulbV2.exe \
 *          mandelbulbV2.c mandelbulb_oclV2.c \
 *          -lpng -lz -lOpenCL -lm
 *
 *  Compile without OpenCL (CPU only):
 *      gcc -O3 -mavx512f -mavx512dq -mavx512bw -mavx512vl -ffast-math \
 *          -o mandelbulbV2.exe mandelbulbV2.c \
 *          -lpng -lz -lm
 *
 *  Compile without AVX-512 (for older CPUs):
 *      gcc -O3 -mavx2 -mfma -ffast-math -DUSE_OPENCL \
 *          -o mandelbulbV2.exe mandelbulbV2.c mandelbulb_oclV2.c \
 *          -lpng -lz -lOpenCL -lm
 *
 *  Requires: libpng, OpenCL (optional), GCC with AVX-512 support
 *
 *  Features: SSAA, soft shadows, improved AO, palettes,
 *            AVX-512 (auto-detected), adaptive step count,
 *            Newton-Raphson refinement, DE safety factor.
 *  DoF is present but broken (see issues).
 * ==========================================================================*/

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <png.h>
#include "mandelbulb_oclV2.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ===========================================================================
 *  Global state
 * ===========================================================================*/

typedef double real;

static int   g_power       = 8;
static int   g_maxIter     = 12;
static int   g_maxSteps    = 256;      /* adaptive -- user can override */
static real  g_bailout     = 8.0;
static real  g_epsScl      = 1e-7;
static real  g_epsMin      = 1e-9;
static real  g_deSafety    = 1.0;      /* scales the DE result (0.5..1.0 safe) */
static int   g_newtonIters = 3;        /* 0 = disable */
static int   g_softShadow  = 1;        /* 1 = on */
static int   g_palette     = 0;        /* 0=copper 1=by-depth 2=by-normal 3=by-shade */
static real  g_dof         = 0.0;      /* aperture radius, 0 = pinhole */
static real  g_lightAz     = 0.0;      /* degrees */
static real  g_lightEl     = 45.0;     /* degrees */

/* Computed light direction (unit) */
static real  g_lx, g_ly, g_lz;

/* ===========================================================================
 *  Vec3 scalar
 * ==========================================================================*/

typedef struct { real x, y, z; } Vec3;
static inline Vec3 V3(real x, real y, real z){ Vec3 v={x,y,z}; return v; }
static inline Vec3 vadd(Vec3 a, Vec3 b){ return V3(a.x+b.x,a.y+b.y,a.z+b.z); }
static inline Vec3 vsub(Vec3 a, Vec3 b){ return V3(a.x-b.x,a.y-b.y,a.z-b.z); }
static inline Vec3 vmul(Vec3 a, real s){ return V3(a.x*s,a.y*s,a.z*s); }
static inline real vdot(Vec3 a, Vec3 b){ return a.x*b.x+a.y*b.y+a.z*b.z; }
static inline real vlen(Vec3 a){ return sqrt(vdot(a,a)); }
static inline Vec3 vnorm(Vec3 a){ real l=vlen(a); return l>1e-30?vmul(a,1.0/l):a; }
static inline Vec3 vcross(Vec3 a, Vec3 b){
    return V3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
static inline real clampd(real x, real a, real b){ return x<a?a:(x>b?b:x); }

/* ===========================================================================
 *  SIMD abstraction  --  same code path for AVX2 and AVX-512
 * ==========================================================================*/

#ifdef __AVX512F__
    typedef __m512d simd_pd;
    #define SIMD_WIDTH 8
    #define simd_set1  _mm512_set1_pd
    #define simd_add   _mm512_add_pd
    #define simd_sub   _mm512_sub_pd
    #define simd_mul   _mm512_mul_pd
    #define simd_div   _mm512_div_pd
    #define simd_sqrt  _mm512_sqrt_pd
    #define simd_fma   _mm512_fmadd_pd
    #define simd_max   _mm512_max_pd
    #define simd_min   _mm512_min_pd
    #define simd_zero  _mm512_setzero_pd()
    #define simd_load  _mm512_loadu_pd
    #define simd_store _mm512_storeu_pd
#else
    typedef __m256d simd_pd;
    #define SIMD_WIDTH 4
    #define simd_set1  _mm256_set1_pd
    #define simd_add   _mm256_add_pd
    #define simd_sub   _mm256_sub_pd
    #define simd_mul   _mm256_mul_pd
    #define simd_div   _mm256_div_pd
    #define simd_sqrt  _mm256_sqrt_pd
    #define simd_fma   _mm256_fmadd_pd
    #define simd_max   _mm256_max_pd
    #define simd_min   _mm256_min_pd
    #define simd_zero  _mm256_setzero_pd()
    #define simd_load  _mm256_loadu_pd
    #define simd_store _mm256_storeu_pd
#endif

/* Boolean ops return 1.0 (true) / 0.0 (false) per lane. */
static inline simd_pd simd_cmp_gt(simd_pd a, simd_pd b){
#ifdef __AVX512F__
    __mmask8 k = _mm512_cmp_pd_mask(a, b, _CMP_GT_OQ);
    return _mm512_mask_blend_pd(k, _mm512_setzero_pd(), _mm512_set1_pd(1.0));
#else
    __m256d m = _mm256_cmp_pd(a, b, _CMP_GT_OQ);
    return _mm256_and_pd(m, _mm256_set1_pd(1.0));
#endif
}
static inline simd_pd simd_cmp_lt(simd_pd a, simd_pd b){
#ifdef __AVX512F__
    __mmask8 k = _mm512_cmp_pd_mask(a, b, _CMP_LT_OQ);
    return _mm512_mask_blend_pd(k, _mm512_setzero_pd(), _mm512_set1_pd(1.0));
#else
    __m256d m = _mm256_cmp_pd(a, b, _CMP_LT_OQ);
    return _mm256_and_pd(m, _mm256_set1_pd(1.0));
#endif
}
static inline simd_pd simd_cmp_ge_zero(simd_pd a){
#ifdef __AVX512F__
    __mmask8 k = _mm512_cmp_pd_mask(a, _mm512_setzero_pd(), _CMP_GE_OQ);
    return _mm512_mask_blend_pd(k, _mm512_setzero_pd(), _mm512_set1_pd(1.0));
#else
    __m256d m = _mm256_cmp_pd(a, _mm256_setzero_pd(), _CMP_GE_OQ);
    return _mm256_and_pd(m, _mm256_set1_pd(1.0));
#endif
}
/* Select: mask > 0 -> b, mask == 0 -> a */
static inline simd_pd simd_select(simd_pd a, simd_pd b, simd_pd mask){
#ifdef __AVX512F__
    __mmask8 k = _mm512_cmp_pd_mask(mask, _mm512_setzero_pd(), _CMP_GT_OQ);
    return _mm512_mask_blend_pd(k, a, b);
#else
    __m256d m = _mm256_cmp_pd(mask, _mm256_setzero_pd(), _CMP_GT_OQ);
    return _mm256_blendv_pd(a, b, m);
#endif
}
static inline int simd_all_true(simd_pd mask){
#ifdef __AVX512F__
    return _mm512_cmp_pd_mask(mask, _mm512_setzero_pd(), _CMP_GT_OQ) == 0xFF;
#else
    return _mm256_movemask_pd(_mm256_cmp_pd(mask, _mm256_setzero_pd(), _CMP_GT_OQ)) == 0xF;
#endif
}
static inline int simd_none_true(simd_pd mask){
#ifdef __AVX512F__
    return _mm512_cmp_pd_mask(mask, _mm512_setzero_pd(), _CMP_GT_OQ) == 0;
#else
    return _mm256_movemask_pd(_mm256_cmp_pd(mask, _mm256_setzero_pd(), _CMP_GT_OQ)) == 0;
#endif
}

typedef struct { simd_pd x, y, z; } Vec3v;

static inline Vec3v SV3(simd_pd x, simd_pd y, simd_pd z){ Vec3v v={x,y,z}; return v; }
static inline Vec3v svadd(Vec3v a, Vec3v b){ return SV3(simd_add(a.x,b.x), simd_add(a.y,b.y), simd_add(a.z,b.z)); }
static inline Vec3v svsub(Vec3v a, Vec3v b){ return SV3(simd_sub(a.x,b.x), simd_sub(a.y,b.y), simd_sub(a.z,b.z)); }
static inline Vec3v svmul(Vec3v a, simd_pd s){ return SV3(simd_mul(a.x,s), simd_mul(a.y,s), simd_mul(a.z,s)); }
static inline simd_pd svdot(Vec3v a, Vec3v b){
    return simd_fma(a.z,b.z, simd_fma(a.y,b.y, simd_mul(a.x,b.x)));
}

/* ===========================================================================
 *  SIMD log2  --  works on both AVX2 and AVX-512
 * ==========================================================================*/

static inline simd_pd simd_log2(simd_pd x)
{
#ifdef __AVX512F__
    /* AVX-512F has proper getmant/getexp */
    simd_pd e = _mm512_getexp_pd(x);
    simd_pd m = _mm512_getmant_pd(x, _MM_MANT_NORM_1_2, _MM_MANT_SIGN_nan);
#else
    /* AVX2: extract exponent with integer shifts */
    __m256i xi = _mm256_srli_epi64(_mm256_castpd_si256(x), 52);
    __m256i ei = _mm256_sub_epi64(xi, _mm256_set1_epi64x(1023));
    __m256i biased = _mm256_add_epi64(ei, _mm256_set1_epi64x(1074));
    __m256i magic  = _mm256_set1_epi64x(0x4330000000000000LL);
    simd_pd e = _mm256_sub_pd(
        _mm256_castsi256_pd(_mm256_or_si256(biased, magic)),
        _mm256_set1_pd(4503599627370496.0));
    e = _mm256_sub_pd(e, _mm256_set1_pd(1074.0));
    const __m256i mantMask = _mm256_set1_epi64x(0x000FFFFFFFFFFFFFLL);
    simd_pd m = _mm256_castsi256_pd(
        _mm256_or_si256(
            _mm256_and_si256(_mm256_castpd_si256(x), mantMask),
            _mm256_set1_epi64x(0x3FF0000000000000LL)));
#endif

    /* normalise m into [sqrt(2)/2, sqrt(2)) */
    simd_pd gt = simd_cmp_gt(m, simd_set1(1.4142135623730951));
    m  = simd_select(m, simd_mul(m, simd_set1(0.5)), gt);
    e  = simd_add(e, gt);   /* add 1.0 where gt == 1 */

    /* log(m) = 2z(1 + z^2/3 + z^4/5 + z^6/7 + z^8/9), z=(m-1)/(m+1) */
    simd_pd one = simd_set1(1.0);
    simd_pd z   = simd_div(simd_sub(m,one), simd_add(m,one));
    simd_pd z2  = simd_mul(z,z);
    simd_pd p   = simd_set1(1.0/9.0);
    p = simd_fma(p, z2, simd_set1(1.0/7.0));
    p = simd_fma(p, z2, simd_set1(1.0/5.0));
    p = simd_fma(p, z2, simd_set1(1.0/3.0));
    p = simd_fma(p, z2, one);
    p = simd_mul(simd_add(z,z), p);
    return simd_fma(p, simd_set1(1.4426950408889634), e);
}

/* ===========================================================================
 *  Distance estimators
 * ==========================================================================*/

static real de_scalar(real cx, real cy, real cz)
{
    real zx=cx, zy=cy, zz=cz, dr=1.0;
    const int n = g_power;
    const real bail2 = g_bailout*g_bailout;
    const real tiny = 1e-30;
    for (int i = 0; i < g_maxIter; i++) {
        real r2 = zx*zx + zy*zy + zz*zz;
        if (r2 > bail2) break;
        real r = sqrt(r2), rs = r>tiny?r:tiny;
        real rxy = sqrt(zx*zx + zy*zy), rxs = rxy>tiny?rxy:tiny;
        real ct = clampd(zz/rs, -1.0, 1.0), st = rxy/rs;
        real cp = zx/rxs, sp = zy/rxs;
        real Tp=1.0,Tc=ct,Up=1.0,Uc=2.0*ct;
        for (int k=2;k<=n;k++){real Tn=2.0*ct*Tc-Tp;real Un=2.0*ct*Uc-Up;Tp=Tc;Tc=Tn;Up=Uc;Uc=Un;}
        real cnt=Tc, snt=st*Up;
        Tp=1.0;Tc=cp;Up=1.0;Uc=2.0*cp;
        for (int k=2;k<=n;k++){real Tn=2.0*cp*Tc-Tp;real Un=2.0*cp*Uc-Up;Tp=Tc;Tc=Tn;Up=Uc;Uc=Un;}
        real cnp=Tc, snp=sp*Up;
        real rn=r; for(int k=1;k<n;k++) rn*=r;
        real rn1=rn/rs;
        dr = (real)n * rn1 * dr + 1.0;
        real stn = rn*snt;
        zx = stn*cnp + cx;
        zy = stn*snp + cy;
        zz = rn *cnt + cz;
    }
    real rf = sqrt(zx*zx+zy*zy+zz*zz);
    rf = rf>tiny?rf:tiny;
    return g_deSafety * 0.5 * log(rf) * rf / dr;
}

static inline simd_pd de_simd(simd_pd cx, simd_pd cy, simd_pd cz)
{
    const int n = g_power;
    simd_pd one = simd_set1(1.0);
    simd_pd tiny = simd_set1(1e-30);
    simd_pd bail2 = simd_set1(g_bailout*g_bailout);
    simd_pd nv = simd_set1((double)n);
    simd_pd minusOne = simd_set1(-1.0);
    simd_pd zx=cx, zy=cy, zz=cz, dr=one;

    for (int i = 0; i < g_maxIter; i++) {
        simd_pd r2 = simd_fma(zx,zx, simd_fma(zy,zy, simd_mul(zz,zz)));
        simd_pd escaped = simd_cmp_gt(r2, bail2);
        if (simd_all_true(escaped)) break;
        simd_pd stillGoing = simd_sub(one, escaped);

        simd_pd r  = simd_sqrt(r2);
        simd_pd rs = simd_max(r, tiny);
        simd_pd rxy2 = simd_fma(zx,zx, simd_mul(zy,zy));
        simd_pd rxy  = simd_sqrt(rxy2);
        simd_pd rxs  = simd_max(rxy, tiny);
        simd_pd ct = simd_max(simd_min(simd_div(zz,rs), one), minusOne);
        simd_pd st = simd_div(rxy,rs);
        simd_pd cp = simd_div(zx,rxs);
        simd_pd sp = simd_div(zy,rxs);

        simd_pd two_ct = simd_add(ct, ct);
        simd_pd Tp=one, Tc=ct, Up=one, Uc=two_ct;
        for (int k=2;k<=n;k++){
            simd_pd Tn = simd_sub(simd_mul(two_ct,Tc), Tp);
            simd_pd Un = simd_sub(simd_mul(two_ct,Uc), Up);
            Tp=Tc;Tc=Tn;Up=Uc;Uc=Un;
        }
        simd_pd cnt = Tc, snt = simd_mul(st, Up);

        simd_pd two_cp = simd_add(cp, cp);
        Tp=one;Tc=cp;Up=one;Uc=two_cp;
        for (int k=2;k<=n;k++){
            simd_pd Tn = simd_sub(simd_mul(two_cp,Tc), Tp);
            simd_pd Un = simd_sub(simd_mul(two_cp,Uc), Up);
            Tp=Tc;Tc=Tn;Up=Uc;Uc=Un;
        }
        simd_pd cnp = Tc, snp = simd_mul(sp, Up);

        simd_pd rn = r;
        for (int k=1;k<n;k++) rn = simd_mul(rn, r);
        simd_pd rn1 = simd_div(rn, rs);
        simd_pd drNew = simd_fma(simd_mul(nv, rn1), dr, one);

        simd_pd stn = simd_mul(rn, snt);
        simd_pd newx = simd_fma(stn, cnp, cx);
        simd_pd newy = simd_fma(stn, snp, cy);
        simd_pd newz = simd_fma(rn, cnt, cz);

        zx = simd_select(zx, newx, stillGoing);
        zy = simd_select(zy, newy, stillGoing);
        zz = simd_select(zz, newz, stillGoing);
        dr = simd_select(dr, drNew, stillGoing);
    }

    simd_pd rf2 = simd_fma(zx,zx, simd_fma(zy,zy, simd_mul(zz,zz)));
    simd_pd rf = simd_max(simd_sqrt(rf2), tiny);
    simd_pd ln_r = simd_mul(simd_log2(rf), simd_set1(0.6931471805599453));
    return simd_mul(simd_set1(g_deSafety * 0.5),
           simd_div(simd_mul(ln_r, rf), dr));
}

/* ===========================================================================
 *  Normals
 * ==========================================================================*/

static Vec3 calc_normal_scalar(Vec3 p)
{
    const real h = 0.0005 * 0.5773502691896258;
    const real nh = -h;
    Vec3 o1 = V3( h, nh, nh), o2 = V3(nh, nh,  h),
         o3 = V3(nh,  h, nh), o4 = V3( h,  h,  h);
    Vec3 n = V3(0,0,0);
    n = vadd(n, vmul(o1, de_scalar(p.x+o1.x, p.y+o1.y, p.z+o1.z)));
    n = vadd(n, vmul(o2, de_scalar(p.x+o2.x, p.y+o2.y, p.z+o2.z)));
    n = vadd(n, vmul(o3, de_scalar(p.x+o3.x, p.y+o3.y, p.z+o3.z)));
    n = vadd(n, vmul(o4, de_scalar(p.x+o4.x, p.y+o4.y, p.z+o4.z)));
    real l = vlen(n); if (l < 1e-30) l = 1e-30;
    return vmul(n, 1.0/l);
}

static Vec3v calc_normal_simd(Vec3v p)
{
    const simd_pd h  = simd_set1(0.0005 * 0.5773502691896258);
    const simd_pd nh = simd_sub(simd_zero, h);
    simd_pd d1 = de_simd(simd_add(p.x,h ), simd_add(p.y,nh), simd_add(p.z,nh));
    simd_pd d2 = de_simd(simd_add(p.x,nh), simd_add(p.y,nh), simd_add(p.z,h ));
    simd_pd d3 = de_simd(simd_add(p.x,nh), simd_add(p.y,h ), simd_add(p.z,nh));
    simd_pd d4 = de_simd(simd_add(p.x,h ), simd_add(p.y,h ), simd_add(p.z,h ));
    simd_pd nx = simd_fma(h, d1, simd_fma(nh, d2, simd_fma(nh, d3, simd_mul(h, d4))));
    simd_pd ny = simd_fma(nh, d1, simd_fma(nh, d2, simd_fma(h, d3, simd_mul(h, d4))));
    simd_pd nz = simd_fma(nh, d1, simd_fma(h, d2, simd_fma(nh, d3, simd_mul(h, d4))));
    Vec3v n = SV3(nx, ny, nz);
    simd_pd l = simd_max(simd_sqrt(svdot(n,n)), simd_set1(1e-30));
    return svmul(n, simd_div(simd_set1(1.0), l));
}

/* ===========================================================================
 *  Ambient occlusion  (8 taps, quadratic falloff)
 * ==========================================================================*/

static real calc_ao_scalar(Vec3 p, Vec3 n)
{
    real occ = 0.0;
    for (int i = 0; i < 8; i++) {
        real h = 0.01 + 0.15 * (real)i * (real)i / 49.0;
        Vec3 q = vadd(p, vmul(n, h));
        real d = de_scalar(q.x, q.y, q.z);
        occ += (h - d) * (1.0 - (real)i / 8.0);
    }
    return clampd(1.0 - 2.5 * occ, 0.0, 1.0);
}

static simd_pd calc_ao_simd(Vec3v p, Vec3v n)
{
    simd_pd occ = simd_zero;
    for (int i = 0; i < 8; i++) {
        simd_pd h = simd_set1(0.01 + 0.15 * (double)i * (double)i / 49.0);
        Vec3v q = svadd(p, svmul(n, h));
        simd_pd d = de_simd(q.x, q.y, q.z);
        simd_pd w = simd_set1(1.0 - (double)i / 8.0);
        occ = simd_fma(simd_sub(h, d), w, occ);
    }
    simd_pd ao = simd_sub(simd_set1(1.0), simd_mul(simd_set1(2.5), occ));
    ao = simd_max(ao, simd_zero);
    ao = simd_min(ao, simd_set1(1.0));
    return ao;
}

/* ===========================================================================
 *  Soft shadow  (returns [0,1], 0 = fully shadowed)
 * ==========================================================================*/

static real trace_shadow_scalar(Vec3 p, Vec3 lig)
{
    real s = 1.0, t = 0.02;
    for (int i = 0; i < 24; i++) {
        Vec3 q = vadd(p, vmul(lig, t));
        real d = de_scalar(q.x, q.y, q.z);
        s = fmin(s, 8.0 * d / t);
        t += fmax(d, 0.02);
        if (s < 0.01) break;
        if (d < 1e-7) return 0.0;
        if (t > 4.0) break;
    }
    return clampd(s, 0.0, 1.0);
}

static simd_pd trace_shadow_simd(Vec3v p, simd_pd lx, simd_pd ly, simd_pd lz)
{
    simd_pd s = simd_set1(1.0);
    simd_pd t = simd_set1(0.02);
    simd_pd alive = simd_set1(1.0);
    for (int i = 0; i < 24; i++) {
        simd_pd qx = simd_fma(lx, t, p.x);
        simd_pd qy = simd_fma(ly, t, p.y);
        simd_pd qz = simd_fma(lz, t, p.z);
        simd_pd d = de_simd(qx, qy, qz);
        simd_pd contrib = simd_div(simd_mul(simd_set1(8.0), d), t);
        simd_pd newS = simd_min(s, contrib);
        s = simd_select(s, newS, alive);
        simd_pd step = simd_max(d, simd_set1(0.02));
        t = simd_add(t, simd_mul(step, alive));
        simd_pd done = simd_cmp_lt(s, simd_set1(0.01));
        simd_pd tooFar = simd_cmp_gt(t, simd_set1(4.0));
        alive = simd_sub(simd_set1(1.0), simd_max(done, tooFar));
        if (simd_none_true(alive)) break;
    }
    return simd_max(simd_min(s, simd_set1(1.0)), simd_zero);
}

/* ===========================================================================
 *  Ray marching + Newton refinement
 * ==========================================================================*/

static real trace_scalar(Vec3 ro, Vec3 rd)
{
    real t = 0.0;
    for (int i = 0; i < g_maxSteps; i++) {
        Vec3 p = vadd(ro, vmul(rd, t));
        real d = de_scalar(p.x, p.y, p.z);
        real eps = fmax(g_epsScl * t, g_epsMin);
        if (d < eps) {
            /* Newton refinement */
            if (g_newtonIters > 0) {
                for (int k = 0; k < g_newtonIters; k++) {
                    Vec3 q = vadd(ro, vmul(rd, t));
                    real dd = de_scalar(q.x, q.y, q.z);
                    if (fabs(dd) < 1e-12) break;
                    Vec3 nn = calc_normal_scalar(q);
                    real denom = vdot(nn, rd);
                    if (fabs(denom) < 1e-8) break;
                    t -= dd / denom;
                }
            }
            return t;
        }
        t += d;
        if (t > 20.0) break;
    }
    return -1.0;
}

static void trace_simd(simd_pd ox, simd_pd oy, simd_pd oz,
                       simd_pd dx, simd_pd dy, simd_pd dz,
                       simd_pd *outT)
{
    simd_pd t      = simd_zero;
    simd_pd tHit   = simd_set1(-1.0);
    simd_pd active = simd_set1(1.0);

    for (int i = 0; i < g_maxSteps; i++) {
        simd_pd px = simd_fma(dx, t, ox);
        simd_pd py = simd_fma(dy, t, oy);
        simd_pd pz = simd_fma(dz, t, oz);
        simd_pd d = de_simd(px, py, pz);
        simd_pd eps = simd_max(simd_mul(simd_set1(g_epsScl), t), simd_set1(g_epsMin));
        simd_pd hit = simd_mul(active, simd_cmp_lt(d, eps));
        tHit = simd_select(tHit, t, hit);
        simd_pd survive = simd_mul(active, simd_sub(simd_set1(1.0), hit));
        t = simd_add(t, simd_mul(d, survive));
        simd_pd over = simd_cmp_gt(t, simd_set1(20.0));
        active = simd_mul(survive, simd_sub(simd_set1(1.0), over));
        if (simd_none_true(active)) break;
    }
    /* Newton refinement for hits */
    if (g_newtonIters > 0) {
        simd_pd tRef = tHit;
        for (int k = 0; k < g_newtonIters; k++) {
            simd_pd px = simd_fma(dx, tRef, ox);
            simd_pd py = simd_fma(dy, tRef, oy);
            simd_pd pz = simd_fma(dz, tRef, oz);
            simd_pd dd = de_simd(px, py, pz);
            Vec3v pv = SV3(px, py, pz);
            Vec3v nv = calc_normal_simd(pv);
            simd_pd denom = simd_fma(nv.x, dx, simd_fma(nv.y, dy, simd_mul(nv.z, dz)));
            simd_pd valid = simd_cmp_gt(simd_max(denom, simd_sub(simd_zero, denom)), simd_set1(1e-8));
            simd_pd step = simd_div(dd, denom);
            step = simd_mul(step, valid);
            tRef = simd_sub(tRef, step);
        }
        /* only apply to lanes with a real hit */
        simd_pd hasHit = simd_cmp_ge_zero(tHit);
        tHit = simd_select(tHit, tRef, hasHit);
    }
    *outT = tHit;
}

/* ===========================================================================
 *  Shading
 * ==========================================================================*/

static void palette_apply(int pal, real t, Vec3 n, real dif,
                          real *baseR, real *baseG, real *baseB)
{
    switch (pal) {
        case 1: {   /* by depth */
            real k = clampd(t / 4.0, 0.0, 1.0);
            *baseR = 0.9 - 0.5*k;
            *baseG = 0.3 + 0.5*k;
            *baseB = 0.1 + 0.7*k;
        } break;
        case 2: {   /* by normal */
            *baseR = 0.5 + 0.5*n.x;
            *baseG = 0.5 + 0.5*n.y;
            *baseB = 0.5 + 0.5*n.z;
        } break;
        case 3: {   /* by shading */
            real k = clampd(dif, 0.0, 1.0);
            *baseR = 0.2 + 0.8*k;
            *baseG = 0.2 + 0.6*k;
            *baseB = 0.2 + 0.4*k;
        } break;
        default: {   /* copper */
            *baseR = 0.92; *baseG = 0.58; *baseB = 0.26;
        } break;
    }
}

static Vec3 shade_scalar(Vec3 ro, Vec3 rd, real t)
{
    if (t < 0) {
        real k = clampd(rd.y*0.5 + 0.5, 0.0, 1.0);
        return V3(0.015 + 0.205*k, 0.020 + 0.360*k, 0.045 + 0.635*k);
    }
    Vec3 p = vadd(ro, vmul(rd, t));
    Vec3 n = calc_normal_scalar(p);
    Vec3 lig = V3(g_lx, g_ly, g_lz);
    real dif = clampd(vdot(n, lig), 0.0, 1.0);
    real sh  = g_softShadow ? trace_shadow_scalar(p, lig) : (dif > 0.0 ? 1.0 : 0.0);
    dif *= sh;
    real amb = clampd(0.5 + 0.5*n.y, 0.0, 1.0);
    real ao  = calc_ao_scalar(p, n);

    real bR, bG, bB;
    palette_apply(g_palette, t, n, dif, &bR, &bG, &bB);

    real lam = amb*0.30*ao + dif*0.90;
    Vec3 col = V3(bR*lam, bG*lam, bB*lam);

    Vec3 hv = vnorm(vsub(lig, rd));
    real spe = pow(clampd(vdot(n, hv), 0.0, 1.0), 48.0);
    col = vadd(col, vmul(V3(1.00, 0.94, 0.82), spe * dif * 0.75));
    real fre = pow(clampd(1.0 + vdot(n, rd), 0.0, 1.0), 4.0);
    col = vadd(col, vmul(V3(0.28, 0.40, 0.62), fre * 0.35));
    return col;
}

static void shade_simd(Vec3v ro, Vec3v rd, simd_pd t, simd_pd hitMask, Vec3v *outCol)
{
    Vec3v p = SV3(simd_fma(t, rd.x, ro.x),
                  simd_fma(t, rd.y, ro.y),
                  simd_fma(t, rd.z, ro.z));
    Vec3v n = calc_normal_simd(p);

    simd_pd one  = simd_set1(1.0);
    simd_pd zero = simd_zero;
    simd_pd half = simd_set1(0.5);
    simd_pd lx = simd_set1(g_lx), ly = simd_set1(g_ly), lz = simd_set1(g_lz);

    simd_pd dif = simd_max(svdot(n, SV3(lx,ly,lz)), zero);

    if (g_softShadow) {
        simd_pd sh = trace_shadow_simd(p, lx, ly, lz);
        dif = simd_mul(dif, sh);
    }

    simd_pd amb = simd_max(simd_min(simd_fma(half, n.y, half), one), zero);
    simd_pd ao  = calc_ao_simd(p, n);

    /* palette -- scalar switch, applied per-lane via stores */
    simd_pd bR = simd_set1(0.92), bG = simd_set1(0.58), bB = simd_set1(0.26);
    if (g_palette == 1) {
        /* by depth */
        simd_pd k = simd_max(simd_min(simd_mul(t, simd_set1(0.25)), one), zero);
        bR = simd_sub(simd_set1(0.9), simd_mul(simd_set1(0.5), k));
        bG = simd_fma(simd_set1(0.5), k, simd_set1(0.3));
        bB = simd_fma(simd_set1(0.7), k, simd_set1(0.1));
    } else if (g_palette == 2) {
        bR = simd_fma(half, n.x, half);
        bG = simd_fma(half, n.y, half);
        bB = simd_fma(half, n.z, half);
    } else if (g_palette == 3) {
        bR = simd_fma(simd_set1(0.8), dif, simd_set1(0.2));
        bG = simd_fma(simd_set1(0.6), dif, simd_set1(0.2));
        bB = simd_fma(simd_set1(0.4), dif, simd_set1(0.2));
    }

    simd_pd lam = simd_fma(amb, simd_mul(simd_set1(0.30), ao),
                           simd_mul(dif, simd_set1(0.90)));
    simd_pd cr = simd_mul(bR, lam);
    simd_pd cg = simd_mul(bG, lam);
    simd_pd cb = simd_mul(bB, lam);

    Vec3v lig = SV3(lx,ly,lz);
    Vec3v hv = svsub(lig, rd);
    simd_pd hlen = simd_sqrt(svdot(hv,hv));
    hv = svmul(hv, simd_div(one, simd_max(hlen, simd_set1(1e-30))));
    simd_pd spe = svdot(n, hv);
    simd_pd s48 = spe;
    for (int i = 0; i < 6; i++) s48 = simd_mul(s48, s48);   /* ^64 -- close enough */
    simd_pd speI = simd_mul(s48, simd_mul(dif, simd_set1(0.75)));
    cr = simd_fma(speI, simd_set1(1.00), cr);
    cg = simd_fma(speI, simd_set1(0.94), cg);
    cb = simd_fma(speI, simd_set1(0.82), cb);

    simd_pd fre = simd_max(simd_min(simd_add(one, svdot(n, rd)), one), zero);
    simd_pd f2 = simd_mul(fre, fre);
    simd_pd fre4 = simd_mul(f2, f2);
    simd_pd freI = simd_mul(fre4, simd_set1(0.35));
    cr = simd_fma(freI, simd_set1(0.28), cr);
    cg = simd_fma(freI, simd_set1(0.40), cg);
    cb = simd_fma(freI, simd_set1(0.62), cb);

    /* sky for misses */
    simd_pd skyK = simd_max(simd_min(simd_fma(rd.y, half, half), one), zero);
    simd_pd sr = simd_fma(skyK, simd_set1(0.205), simd_set1(0.015));
    simd_pd sg = simd_fma(skyK, simd_set1(0.360), simd_set1(0.020));
    simd_pd sb = simd_fma(skyK, simd_set1(0.635), simd_set1(0.045));
    cr = simd_select(sr, cr, hitMask);
    cg = simd_select(sg, cg, hitMask);
    cb = simd_select(sb, cb, hitMask);
    outCol->x = cr;
    outCol->y = cg;
    outCol->z = cb;
}

/* ===========================================================================
 *  Camera & job dispatch
 * ==========================================================================*/

typedef struct { int y0, y1; } Job;

static int      g_width = 640, g_height = 480;
static int      g_ss    = 1;              /* supersample factor */
static int      g_forceScalar = 0;
static uint8_t *g_img   = NULL;           /* supersampled buffer */
static int      g_sw = 0, g_sh = 0;       /* supersampled dims */

static Vec3  g_camPos, g_camFwd, g_camRight, g_camUp;
static real  g_tanHalfFov;
static real  g_crx, g_cry, g_crz, g_cux, g_cuy, g_cuz, g_cfx, g_cfy, g_cfz;

static inline real gamma_22(real x){ if(x<0)x=0; if(x>1)x=1; return pow(x,1.0/2.2); }
static inline uint8_t tone_store(real lin){ real r=lin/(1.0+lin); return (uint8_t)(gamma_22(r)*255.0+0.5); }

/* Deterministic per-pixel hash for DoF jitter */
static inline void dof_jitter(int px, int py, real *ox, real *oy)
{
    if (g_dof <= 0.0) { *ox = 0.0; *oy = 0.0; return; }
    uint32_t h = (uint32_t)px * 73856093u ^ (uint32_t)py * 19349663u;
    h = h * 1664525u + 1013904223u;
    real r1 = (real)((h >> 8) & 0xFFFF) / 65535.0;
    h = h * 1664525u + 1013904223u;
    real r2 = (real)((h >> 8) & 0xFFFF) / 65535.0;
    real r  = sqrt(r1) * g_dof;
    real th = r2 * 2.0 * M_PI;
    *ox = r * cos(th);
    *oy = r * sin(th);
}

static DWORD WINAPI workerProc(LPVOID param)
{
    Job *job = (Job *)param;
    const real aspect = (real)g_sw / (real)g_sh;

    for (int y = job->y0; y < job->y1; y++) {
        uint8_t *row = g_img + (size_t)y * (size_t)g_sw * 3u;

        if (!g_forceScalar && (SIMD_WIDTH == 4 || SIMD_WIDTH == 8)) {
            int x = 0;
            for (; x + SIMD_WIDTH <= g_sw; x += SIMD_WIDTH) {
                /* build per-lane coords, DoF jitter, ray dirs */
                double uu[SIMD_WIDTH], vv[SIMD_WIDTH];
                double jox[SIMD_WIDTH], joy[SIMD_WIDTH];
                for (int k = 0; k < SIMD_WIDTH; k++) {
                    int sx = x + k;
                    real fx = ((real)sx + 0.5) / (real)g_sw;
                    real fy = ((real)y + 0.5) / (real)g_sh;
                    uu[k] = (2.0*fx - 1.0) * aspect;
                    vv[k] = 1.0 - 2.0*fy;
                    real ox_, oy_;
                    dof_jitter(sx, y, &ox_, &oy_);
                    jox[k] = ox_; joy[k] = oy_;
                }
                simd_pd u  = simd_load(uu);
                simd_pd v  = simd_load(vv);
                simd_pd jx = simd_load(jox);
                simd_pd jy = simd_load(joy);

                simd_pd ox = simd_add(simd_set1(g_camPos.x),
                                      simd_fma(simd_set1(g_camRight.x), jx,
                                               simd_mul(simd_set1(g_camUp.x), jy)));
                simd_pd oy = simd_add(simd_set1(g_camPos.y),
                                      simd_fma(simd_set1(g_camRight.y), jx,
                                               simd_mul(simd_set1(g_camUp.y), jy)));
                simd_pd oz = simd_add(simd_set1(g_camPos.z),
                                      simd_fma(simd_set1(g_camRight.z), jx,
                                               simd_mul(simd_set1(g_camUp.z), jy)));

                simd_pd rdx = simd_fma(simd_set1(g_crx), u,
                              simd_fma(simd_set1(g_cux), v, simd_set1(g_cfx)));
                simd_pd rdy = simd_fma(simd_set1(g_cry), u,
                              simd_fma(simd_set1(g_cuy), v, simd_set1(g_cfy)));
                simd_pd rdz = simd_fma(simd_set1(g_crz), u,
                              simd_fma(simd_set1(g_cuz), v, simd_set1(g_cfz)));
                simd_pd len2 = simd_fma(rdx,rdx, simd_fma(rdy,rdy, simd_mul(rdz,rdz)));
                simd_pd invl = simd_div(simd_set1(1.0), simd_sqrt(len2));
                rdx = simd_mul(rdx, invl);
                rdy = simd_mul(rdy, invl);
                rdz = simd_mul(rdz, invl);

                simd_pd t;
                trace_simd(ox, oy, oz, rdx, rdy, rdz, &t);

                simd_pd hitMask = simd_cmp_ge_zero(t);
                simd_pd tSafe = simd_select(simd_set1(1.0), t, hitMask);
                Vec3v ro = SV3(ox,oy,oz), rd = SV3(rdx,rdy,rdz);
                Vec3v col;
                shade_simd(ro, rd, tSafe, hitMask, &col);

                double rr[SIMD_WIDTH], gg[SIMD_WIDTH], bb[SIMD_WIDTH];
                simd_store(rr, col.x); simd_store(gg, col.y); simd_store(bb, col.z);
                for (int k = 0; k < SIMD_WIDTH; k++) {
                    row[(x+k)*3+0] = tone_store(rr[k]);
                    row[(x+k)*3+1] = tone_store(gg[k]);
                    row[(x+k)*3+2] = tone_store(bb[k]);
                }
            }
            for (; x < g_sw; x++) {
                real fx = ((real)x + 0.5) / (real)g_sw;
                real fy = ((real)y + 0.5) / (real)g_sh;
                real uu_ = (2.0*fx-1.0)*aspect, vv_ = 1.0-2.0*fy;
                real jox_, joy_; dof_jitter(x, y, &jox_, &joy_);
                Vec3 ro = vadd(g_camPos, vadd(vmul(g_camRight,jox_), vmul(g_camUp,joy_)));
                Vec3 rd = vnorm(vadd(g_camFwd, vadd(vmul(g_camRight,uu_*g_tanHalfFov), vmul(g_camUp,vv_*g_tanHalfFov))));
                real tt = trace_scalar(ro, rd);
                Vec3 col = shade_scalar(ro, rd, tt);
                row[x*3+0] = tone_store(col.x);
                row[x*3+1] = tone_store(col.y);
                row[x*3+2] = tone_store(col.z);
            }
        } else {
            for (int x = 0; x < g_sw; x++) {
                real fx = ((real)x + 0.5) / (real)g_sw;
                real fy = ((real)y + 0.5) / (real)g_sh;
                real uu_ = (2.0*fx-1.0)*aspect, vv_ = 1.0-2.0*fy;
                real jox_, joy_; dof_jitter(x, y, &jox_, &joy_);
                Vec3 ro = vadd(g_camPos, vadd(vmul(g_camRight,jox_), vmul(g_camUp,joy_)));
                Vec3 rd = vnorm(vadd(g_camFwd, vadd(vmul(g_camRight,uu_*g_tanHalfFov), vmul(g_camUp,vv_*g_tanHalfFov))));
                real tt = trace_scalar(ro, rd);
                Vec3 col = shade_scalar(ro, rd, tt);
                row[x*3+0] = tone_store(col.x);
                row[x*3+1] = tone_store(col.y);
                row[x*3+2] = tone_store(col.z);
            }
        }
    }
    return 0;
}

/* Downsample SSAA buffer to final image */
static void downsample_ssaa(const uint8_t *src, int sw, int sh, int ss,
                            uint8_t *dst, int w, int h)
{
    const int n = ss * ss;
    for (int y = 0; y < h; y++) {
        uint8_t *drow = dst + (size_t)y * (size_t)w * 3u;
        for (int x = 0; x < w; x++) {
            int r = 0, g = 0, b = 0;
            for (int sy = 0; sy < ss; sy++) {
                const uint8_t *srow = src + (size_t)(y*ss + sy) * (size_t)sw * 3u;
                for (int sx = 0; sx < ss; sx++) {
                    const uint8_t *p = srow + (size_t)(x*ss + sx) * 3u;
                    r += p[0]; g += p[1]; b += p[2];
                }
            }
            drow[x*3+0] = (uint8_t)(r / n);
            drow[x*3+1] = (uint8_t)(g / n);
            drow[x*3+2] = (uint8_t)(b / n);
        }
    }
}

/* ===========================================================================
 *  PNG writer
 * ==========================================================================*/

static int writePNG(const char *path, int w, int h, const uint8_t *topDown)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) { fprintf(stderr, "Cannot open %s\n", path); return 0; }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png) { fclose(fp); return 0; }
    png_infop info = png_create_info_struct(png);
    if (!info) { png_destroy_write_struct(&png, NULL); fclose(fp); return 0; }
    if (setjmp(png_jmpbuf(png))) { png_destroy_write_struct(&png, &info); fclose(fp); return 0; }
    png_init_io(png, fp);
    png_set_IHDR(png, info, (png_uint_32)w, (png_uint_32)h, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_compression_level(png, 6);
    png_write_info(png, info);
    png_bytep rowbuf = (png_bytep)malloc((size_t)w * 3u);
    if (!rowbuf) { png_destroy_write_struct(&png, &info); fclose(fp); return 0; }
    for (int y = 0; y < h; y++) {
        memcpy(rowbuf, topDown + (size_t)y*(size_t)w*3u, (size_t)w*3u);
        png_write_row(png, rowbuf);
    }
    free(rowbuf);
    png_write_end(png, NULL);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return 1;
}

/* ===========================================================================
 *  Menu
 * ==========================================================================*/

static void print_menu(FILE *f, const char *prog)
{
    const char *simdName =
#ifdef __AVX512F__
        "AVX-512 (8-wide SIMD)";
#else
        "AVX2 (4-wide SIMD)";
#endif
    fprintf(f,
"\n"
"==================================================================\n"
"  MANDELBULB V2  --  %s\n"
"==================================================================\n"
"\n"
"  Usage:  %s [options]\n"
"\n"
"  Output\n"
"    -o <file.png>       output PNG              (mandelbulb.png)\n"
"    -w <int>            width                    (640)\n"
"    -h <int>            height                   (480)\n"
"\n"
"  Quality\n"
"    -s <int>            supersample 1..4         (1)\n"
"    -maxsteps <int>     max raymarch steps       (256)\n"
"    -eps <real>         DE hit threshold scale   (1e-7)\n"
"    -de-safety <real>   DE conservative factor   (1.0)\n"
"    -newton <int>       Newton refinement iters  (3, 0=off)\n"
"\n"
"  Lighting / Look\n"
"    -palette <int>      0=copper 1=depth 2=normal 3=shade  (0)\n"
"    -softshadow <0|1>   enable soft shadows      (1)\n"
"    -light-az <deg>     light azimuth            (0)\n"
"    -light-el <deg>     light elevation          (45)\n"
"\n"
"  Fractal\n"
"    -p <int>            Mandelbulb power         (8)\n"
"    -i <int>            max DE iterations        (12)\n"
"\n"
"  Camera\n"
"    -f <deg>            vertical FOV             (45)\n"
"    -d <real>           camera distance          (2.6)\n"
"    -az <deg>           azimuth                  (0)\n"
"    -el <deg>           elevation                (0)\n"
"\n"
"  CPU\n"
"    -t <int>            thread count             (all)\n"
"    -scalar             force scalar CPU path\n"
"    -debug              print DE probe, exit\n"
"\n"
"  GPU (OpenCL)\n"
"    -gpu                float kernel\n"
"    -gpu-d              double kernel\n"
"    -gpu-list           list devices, exit\n"
"\n"
"  Misc\n"
"    -?, --help          this menu\n"
"\n", simdName, prog);
}

/* ===========================================================================
 *  main
 * ==========================================================================*/

int main(int argc, char **argv)
{
    if (argc < 2) { print_menu(stdout, argv[0]); return 0; }

    const char *outPath = "mandelbulb.png";
    int  nThreads = 0;
    real fovDeg = 45.0, camDist = 2.6, azDeg = 0.0, elDeg = 0.0;
    int  useGpu = 0, gpuDouble = 0, gpuList = 0;

    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if      (!strcmp(a,"-w")  && i+1<argc) g_width   = atoi(argv[++i]);
        else if (!strcmp(a,"-h")  && i+1<argc) g_height  = atoi(argv[++i]);
        else if (!strcmp(a,"-o")  && i+1<argc) outPath   = argv[++i];
        else if (!strcmp(a,"-t")  && i+1<argc) nThreads  = atoi(argv[++i]);
        else if (!strcmp(a,"-s")  && i+1<argc) g_ss      = atoi(argv[++i]);
        else if (!strcmp(a,"-p")  && i+1<argc) g_power   = atoi(argv[++i]);
        else if (!strcmp(a,"-i")  && i+1<argc) g_maxIter = atoi(argv[++i]);
        else if (!strcmp(a,"-f")  && i+1<argc) fovDeg    = atof(argv[++i]);
        else if (!strcmp(a,"-d")  && i+1<argc) camDist   = atof(argv[++i]);
        else if (!strcmp(a,"-az") && i+1<argc) azDeg     = atof(argv[++i]);
        else if (!strcmp(a,"-el") && i+1<argc) elDeg     = atof(argv[++i]);
        else if (!strcmp(a,"-scalar")) g_forceScalar = 1;
        else if (!strcmp(a,"-gpu"))   useGpu = 1;
        else if (!strcmp(a,"-gpu-d")) { useGpu = 1; gpuDouble = 1; }
        else if (!strcmp(a,"-gpu-list")) gpuList = 1;
        else if (!strcmp(a,"-maxsteps") && i+1<argc) g_maxSteps = atoi(argv[++i]);
        else if (!strcmp(a,"-eps")      && i+1<argc) { g_epsScl = atof(argv[++i]); g_epsMin = g_epsScl*1e-2; }
        else if (!strcmp(a,"-de-safety")&& i+1<argc) g_deSafety = atof(argv[++i]);
        else if (!strcmp(a,"-newton")   && i+1<argc) g_newtonIters = atoi(argv[++i]);
        else if (!strcmp(a,"-palette")  && i+1<argc) g_palette = atoi(argv[++i]);
        else if (!strcmp(a,"-softshadow")&& i+1<argc) g_softShadow = atoi(argv[++i]);
        else if (!strcmp(a,"-light-az") && i+1<argc) g_lightAz = atof(argv[++i]);
        else if (!strcmp(a,"-light-el") && i+1<argc) g_lightEl = atof(argv[++i]);
        else if (!strcmp(a,"-dof")      && i+1<argc) g_dof = atof(argv[++i]);
        else if (!strcmp(a,"-?") || !strcmp(a,"--help")) { print_menu(stdout, argv[0]); return 0; }
        else { fprintf(stderr, "Unknown: %s\n", a); print_menu(stderr, argv[0]); return 1; }
    }

    if (g_width < 16) g_width = 16;
    if (g_height < 16) g_height = 16;
    if (g_ss < 1) g_ss = 1;
    if (g_ss > 4) g_ss = 4;
    if (g_power < 2) g_power = 2;
    if (g_power > 32) g_power = 32;
    if (g_maxIter < 1) g_maxIter = 1;
    if (g_maxIter > 64) g_maxIter = 64;
    if (g_maxSteps < 16) g_maxSteps = 16;
    if (g_maxSteps > 2048) g_maxSteps = 2048;
    if (g_newtonIters < 0) g_newtonIters = 0;
    if (g_newtonIters > 8) g_newtonIters = 8;
    if (g_palette < 0) g_palette = 0;
    if (g_palette > 3) g_palette = 3;
    if (g_dof < 0.0) g_dof = 0.0;

    /* Light direction from azimuth/elevation */
    {
        real laz = g_lightAz * (real)M_PI / 180.0;
        real lel = g_lightEl * (real)M_PI / 180.0;
        g_lx = cos(lel) * sin(laz);
        g_ly = sin(lel);
        g_lz = cos(lel) * cos(laz);
    }

    /* Camera */
    {
        real az = azDeg * (real)M_PI / 180.0;
        real el = elDeg * (real)M_PI / 180.0;
        g_camPos = V3(camDist*cos(el)*sin(az), camDist*sin(el), camDist*cos(el)*cos(az));
        Vec3 target = V3(0,0,0), up = V3(0,1,0);
        g_camFwd   = vnorm(vsub(target, g_camPos));
        g_camRight = vnorm(vcross(g_camFwd, up));
        g_camUp    = vcross(g_camRight, g_camFwd);
        g_tanHalfFov = tan(fovDeg * 0.5 * (real)M_PI / 180.0);
        g_crx = g_camRight.x*g_tanHalfFov;
        g_cry = g_camRight.y*g_tanHalfFov;
        g_crz = g_camRight.z*g_tanHalfFov;
        g_cux = g_camUp.x   *g_tanHalfFov;
        g_cuy = g_camUp.y   *g_tanHalfFov;
        g_cuz = g_camUp.z   *g_tanHalfFov;
        g_cfx = g_camFwd.x; g_cfy = g_camFwd.y; g_cfz = g_camFwd.z;
    }

    if (gpuList) { ocl_list_devices(); return 0; }

    SYSTEM_INFO si; GetSystemInfo(&si);
    int nCPU = (int)si.dwNumberOfProcessors; if (nCPU < 1) nCPU = 1;
    if (nThreads <= 0) nThreads = nCPU;

    g_sw = g_width  * g_ss;
    g_sh = g_height * g_ss;
    size_t imgBytes = (size_t)g_sw * (size_t)g_sh * 3u;
    g_img = (uint8_t*)malloc(imgBytes);
    if (!g_img) { fprintf(stderr, "OOM\n"); return 1; }

    /* ---- banner ---- */
    printf("Mandelbulb V2  %dx%d  ss=%d -> %dx%d  power=%d  iter=%d\n",
           g_width, g_height, g_ss, g_sw, g_sh, g_power, g_maxIter);
    if (useGpu) {
        printf("Path    : OpenCL GPU (%s kernel)\n", gpuDouble ? "double" : "float");
    } else {
        printf("Path    : %s\n",
               g_forceScalar ? "scalar (reference)" :
#ifdef __AVX512F__
               "AVX-512 (8-wide SIMD)"
#else
               "AVX2 (4-wide SIMD)"
#endif
               );
    }
    if (!useGpu) printf("Threads : %d / %d logical cores\n", nThreads, nCPU);
    printf("Camera  : dist=%.3f az=%.1f el=%.1f fov=%.1f\n", camDist, azDeg, elDeg, fovDeg);
    printf("Light   : az=%.1f el=%.1f\n", g_lightAz, g_lightEl);
    printf("Palette : %d   DoF=%.4f   SoftShadow=%d   Newton=%d\n",
           g_palette, g_dof, g_softShadow, g_newtonIters);
    printf("Rendering...\n"); fflush(stdout);

    LARGE_INTEGER freq, t0, t1;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&t0);

    /* ---- GPU path ---- */
    if (useGpu) {
        int ok = ocl_renderV2(g_img, g_sw, g_sh,
                              g_camPos.x, g_camPos.y, g_camPos.z,
                              g_crx, g_cry, g_crz, g_cux, g_cuy, g_cuz,
                              g_cfx, g_cfy, g_cfz,
                              g_power, g_maxIter, g_maxSteps,
                              g_bailout, g_epsScl, g_epsMin, g_deSafety,
                              g_newtonIters, g_softShadow,
                              g_palette, g_dof, g_lx, g_ly, g_lz,
                              gpuDouble);
        if (ok) {
            uint8_t *final = g_img;
            if (g_ss > 1) {
                final = (uint8_t*)malloc((size_t)g_width*(size_t)g_height*3u);
                downsample_ssaa(g_img, g_sw, g_sh, g_ss, final, g_width, g_height);
            }
            QueryPerformanceCounter(&t1);
            double ms = 1000.0*(double)(t1.QuadPart - t0.QuadPart)/(double)freq.QuadPart;
            if (!writePNG(outPath, g_width, g_height, final)) {
                fprintf(stderr, "PNG write failed\n"); return 1;
            }
            printf("Done in %.3f s  ->  %s  (GPU)\n", ms/1000.0, outPath);
            if (final != g_img) free(final);
            free(g_img);
            return 0;
        }
        fprintf(stderr, "GPU render failed; falling back to CPU.\n");
    }

    /* ---- CPU path ---- */
    {
        HANDLE *handles = (HANDLE*)malloc(sizeof(HANDLE)*(size_t)nThreads);
        Job    *jobs    = (Job*)malloc(sizeof(Job)*(size_t)nThreads);
        int rowsPer = (g_sh + nThreads - 1) / nThreads;
        for (int i = 0; i < nThreads; i++) {
            jobs[i].y0 = i*rowsPer;
            jobs[i].y1 = jobs[i].y0 + rowsPer;
            if (jobs[i].y1 > g_sh) jobs[i].y1 = g_sh;
            if (jobs[i].y0 > g_sh) jobs[i].y0 = g_sh;
            handles[i] = CreateThread(NULL, 0, workerProc, &jobs[i], 0, NULL);
            if (!handles[i]) { workerProc(&jobs[i]); handles[i] = NULL; }
        }
        for (int i = 0; i < nThreads; i++)
            if (handles[i]) WaitForSingleObject(handles[i], INFINITE);

        uint8_t *final = g_img;
        if (g_ss > 1) {
            final = (uint8_t*)malloc((size_t)g_width*(size_t)g_height*3u);
            downsample_ssaa(g_img, g_sw, g_sh, g_ss, final, g_width, g_height);
        }
        QueryPerformanceCounter(&t1);
        double ms = 1000.0*(double)(t1.QuadPart - t0.QuadPart)/(double)freq.QuadPart;
        if (!writePNG(outPath, g_width, g_height, final)) {
            fprintf(stderr, "PNG write failed\n"); return 1;
        }
        printf("Done in %.3f s  ->  %s\n", ms/1000.0, outPath);
        if (final != g_img) free(final);
        for (int i = 0; i < nThreads; i++) if (handles[i]) CloseHandle(handles[i]);
        free(handles); free(jobs);
    }
    free(g_img);
    return 0;
}