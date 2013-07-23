#include "timer.h"
#include "allocation.h"
#include "interpolate.h"
#include "padding_aware.h"
#include "phase_shift.h"
#include "naive.h"
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>

typedef enum
{
  INTERLEAVED,
  SPLIT
} storage_layout_t;

typedef struct
{
  storage_layout_t layout;
  union
  {
    fftw_complex *interleaved;
    struct
    {
      double *real;
      double *imag;
    } split;
  } coarse;
  union
  {
    fftw_complex *interleaved;
    struct
    {
      double *real;
      double *imag;
    } split;
  } fine;
} storage_t;

static void allocate_storage(storage_t *storage, storage_layout_t layout, int size)
{
  assert(storage != NULL);
  storage->layout = layout;

  switch(layout)
  {
    case INTERLEAVED:
      storage->coarse.interleaved = rs_alloc_complex(size);
      storage->fine.interleaved = rs_alloc_complex(8 * size);
      break;
    case SPLIT:
      storage->coarse.split.real = rs_alloc_real(size);
      storage->coarse.split.imag = rs_alloc_real(size);
      storage->fine.split.real = rs_alloc_real(8 * size);
      storage->fine.split.imag = rs_alloc_real(8 * size);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", layout);
      exit(EXIT_FAILURE);
  }
}

static void free_storage(storage_t *storage)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      rs_free(storage->coarse.interleaved);
      rs_free(storage->fine.interleaved);
      break;
    case SPLIT:
      rs_free(storage->coarse.split.real);
      rs_free(storage->coarse.split.imag);
      rs_free(storage->fine.split.real);
      rs_free(storage->fine.split.imag);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

static void set_element(storage_t *storage, int offset, fftw_complex value)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      storage->coarse.interleaved[offset] = value;
      break;
    case SPLIT:
      storage->coarse.split.real[offset] = creal(value);
      storage->coarse.split.imag[offset] = cimag(value);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

static void execute_interpolate(storage_t *storage, interpolate_plan plan)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      interpolate_execute_interleaved(plan, storage->coarse.interleaved, storage->fine.interleaved);
      break;
    case SPLIT:
      interpolate_execute_split(plan, storage->coarse.split.real, storage->coarse.split.imag, storage->fine.split.real, storage->fine.split.imag);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

static fftw_complex get_element(storage_t *storage, int offset)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      return storage->fine.interleaved[offset];
    case SPLIT:
      return storage->fine.split.real[offset] + storage->fine.split.imag[offset] * I;
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

static const double pi = 3.14159265358979323846;

static double test_function(double x, double y, double z)
{
  return sin(2 * x) + cos(3 * y) + sin(4 * z);
}

static void perform_timing(plan_constructor_t constructor,
  storage_t *storage, const int z_width, const int y_width, const int x_width)
{
  time_point_t begin_resample, end_resample, begin_plan, end_plan;

  time_point_save(&begin_plan);
  interpolate_plan plan = constructor(z_width, y_width, x_width, 0);
  time_point_save(&end_plan);


  for(int z=0; z < z_width; ++z)
  {
    for(int y=0; y < y_width; ++y)
    {
      for(int x=0; x < x_width; ++x)
      {
        const int offset = z*y_width*x_width + y*x_width + x;
        const double x_pos = (x * 2.0 * pi)/x_width;
        const double y_pos = (y * 2.0 * pi)/y_width;
        const double z_pos = (z * 2.0 * pi)/z_width;

        set_element(storage, offset, test_function(x_pos, y_pos, z_pos));
      }
    }
  }

  time_point_save(&begin_resample);
  execute_interpolate(storage, plan);
  time_point_save(&end_resample);

  double abs_val = 0.0;

  for(int z=0; z < 2 * z_width; ++z)
  {
    for(int y=0; y < 2 * y_width; ++y)
    {
      for(int x=0; x < 2 * x_width; ++x)
      {
        const int offset = 4*z*y_width*x_width + 2*y*x_width + x;
        const double x_pos = (x * pi)/x_width;
        const double y_pos = (y * pi)/y_width;
        const double z_pos = (z * pi)/z_width;

        const double expected = test_function(x_pos, y_pos, z_pos);
        abs_val += cabs(get_element(storage, offset) - expected);
      }
    }
  }

  printf("Interpolation variant: %s\n", interpolate_get_name(plan));
  printf("Problem size: %d x %d x %d\n", x_width, y_width, z_width);
  interpolate_print_timings(plan);
  printf("Planning time: %f\n", time_point_delta(&begin_plan, &end_plan));
  printf("Execution time: %f\n", time_point_delta(&begin_resample, &end_resample));
  printf("Delta: %f\n", abs_val);

  interpolate_destroy_plan(plan);
}

int main(int argc, char **argv)
{
  int z = 75, y = 75, x = 75;
  storage_t storage;

  allocate_storage(&storage, INTERLEAVED, x * y * z);
  perform_timing(interpolate_plan_3d_naive_interleaved, &storage, z, y, x);
  printf("\n");
  perform_timing(interpolate_plan_3d_padding_aware_interleaved, &storage, z, y, x);
  printf("\n");
  perform_timing(interpolate_plan_3d_phase_shift_interleaved, &storage, z, y, x);
  free_storage(&storage);

  printf("\n\n");
  allocate_storage(&storage, SPLIT, x * y * z);
  perform_timing(interpolate_plan_3d_naive_split, &storage, z, y, x);
  printf("\n");
  perform_timing(interpolate_plan_3d_padding_aware_split, &storage, z, y, x);
  printf("\n");
  perform_timing(interpolate_plan_3d_phase_shift_split, &storage, z, y, x);
  free_storage(&storage);

  return EXIT_SUCCESS;
}
