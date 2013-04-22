#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include <fftw3.h>

typedef struct
{
  int dims[3];
  int strides[3];
  fftw_plan dfts[3];
  fftw_plan idfts[3];
  fftw_complex *rotations[3];
} interpolate_plan_s;

typedef interpolate_plan_s *interpolate_plan;

interpolate_plan plan_interpolate_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out);
void interpolate_execute(const interpolate_plan plan, fftw_complex *in, fftw_complex *out);

#endif
