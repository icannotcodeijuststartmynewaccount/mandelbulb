#ifndef MANDELBULB_OCL_V2_H
#define MANDELBULB_OCL_V2_H
#include <stdint.h>

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
                 int use_double);

int ocl_list_devices(void);

#endif