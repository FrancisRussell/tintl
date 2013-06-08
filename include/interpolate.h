#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "timer.h"
#include <complex.h>
#include <fftw3.h>

typedef struct
{
  int stage[2][3];
  int interpolation;
  int dims[3];
  int strides[3];

  fftw_plan dfts[3];
  fftw_plan dfts_staged[3];

  fftw_plan idfts[3];
  fftw_plan idfts_staged[3];

  fftw_complex *rotations[3];

  time_point_t before_expand2;
  time_point_t before_expand1;
  time_point_t before_expand0;
  time_point_t before_gather;
  time_point_t end;
} interpolate_plan_s;

typedef interpolate_plan_s *interpolate_plan;

interpolate_plan plan_interpolate_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int flags);
void interpolate_execute(const interpolate_plan plan, fftw_complex *in, fftw_complex *out);

interpolate_plan plan_interpolate_3d_split(int n0, int n1, int n2, int flags);
void interpolate_execute_split(const interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);

interpolate_plan plan_interpolate_3d_split_product(int n0, int n1, int n2, int flags);
void interpolate_execute_split_product(const interpolate_plan plan, double *rin, double *iin, double *out);

void interpolate_print_timings(const interpolate_plan plan);
void interpolate_destroy_plan(interpolate_plan plan);

#endif
