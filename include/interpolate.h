#ifndef INTERPOLATE_H
#define INTERPOLATE_H

#include "timer.h"
#include <complex.h>
#include <fftw3.h>

/// \file
/// Functions for executing interpolation plans.

/// Data structure common to all different interpolation implementations.
///
/// Implementations populate this struct with a pointer to
/// implementation specific data and function pointers / that this data as
/// a parameter.

typedef struct
{
  /// Reference count
  int ref_cnt;
  /// Pointer to implementation-specific plan information
  void *detail;

  const char *(*get_name)(const void *detail);
  void (*execute_interleaved)(const void *detail, fftw_complex *in, fftw_complex *out);
  void (*execute_split)(const void *detail, double *rin, double *iin, double *rout, double *iout);
  void (*execute_split_product)(const void *detail, double *rin, double *iin, double *out);
  void (*print_timings)(const void *detail);
  void (*destroy_detail)(void* detail);
} interpolate_plan_s;

/// Typedef for client use.
typedef interpolate_plan_s *interpolate_plan;

/// Returns name of a plan
const char *interpolate_get_name(const interpolate_plan plan);

/// Executes an interleaved plan
void interpolate_execute_interleaved(const interpolate_plan plan, fftw_complex *in, fftw_complex *out);

/// Executes a split plan
void interpolate_execute_split(const interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);

/// Executes a split-product plan
void interpolate_execute_split_product(const interpolate_plan plan, double *rin, double *iin, double *out);

/// Prints implementation-specific timing details to standard output
void interpolate_print_timings(const interpolate_plan plan);

/// Destroys the given implementation specific part of a plan
void interpolate_destroy_plan(interpolate_plan plan);

/// Construct the best-performing interleaved interpolation plan from
/// multiple implementations.
interpolate_plan interpolate_plan_3d_interleaved_best(int n0, int n1, int n2, int flags);

/// Construct the best-performing split interpolation plan from
/// multiple implementations.
interpolate_plan interpolate_plan_3d_split_best(int n0, int n1, int n2, int flags);

/// Construct the best-performing split-product interpolation plan from
/// multiple implementations.
interpolate_plan interpolate_plan_3d_split_product_best(int n0, int n1, int n2, int flags);

#endif
