#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "timer.h"
#include <complex.h>
#include <fftw3.h>

typedef struct
{
  void *detail;
  const char *(*get_name)(const void *detail);
  void (*execute_interleaved)(const void *detail, fftw_complex *in, fftw_complex *out);
  void (*execute_split)(const void *detail, double *rin, double *iin, double *rout, double *iout);
  void (*execute_split_product)(const void *detail, double *rin, double *iin, double *out);
  void (*print_timings)(const void *detail);
  void (*destroy_detail)(void* detail);

} interpolate_plan_s;

typedef interpolate_plan_s *interpolate_plan;

const char *interpolate_get_name(const interpolate_plan plan);
void interpolate_execute_interleaved(const interpolate_plan plan, fftw_complex *in, fftw_complex *out);
void interpolate_execute_split(const interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);
void interpolate_execute_split_product(const interpolate_plan plan, const void *detail, double *rin, double *iin, double *out);
void interpolate_print_timings(const interpolate_plan plan);
void interpolate_destroy_plan(interpolate_plan plan);

#endif
