#ifndef INTERPOLATE_INTERFACE_H
#define INTERPOLATE_INTERFACE_H

#include <complex.h>
#include <fftw3.h>

typedef struct {
  const char *(*get_name)(void);
  void *(*plan)(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int flags);
  void (*execute)(const void *plan, fftw_complex *in, fftw_complex *out);
  void (*print_timings)(const void *plan);
  void (*destroy_plan)(void* plan);
} interpolate_interface;

#endif
