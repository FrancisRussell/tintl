#include "timer.h"
#include "allocation.h"
#include "interpolate.h"
#include "padding_aware.h"
#include "phase_shift.h"
#include "naive.h"
#include "naive_cuda.h"
#include "padding_aware_cuda.h"
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

static const double pi = 3.14159265358979323846;

typedef enum
{
  INTERLEAVED,
  SPLIT
} storage_layout_t;

typedef struct
{
  storage_layout_t layout;
  fftw_complex *interleaved;
  struct
  {
    double *real;
    double *imag;
  } split;
} storage_t;

static const char *layout_name(const storage_layout_t layout)
{
  switch(layout)
  {
    case INTERLEAVED:
      return "interleaved";
    case SPLIT:
        return "split";
    default:
        return "unknown";
  }
}

static void storage_allocate(storage_t *storage, storage_layout_t layout, size_t size)
{
  assert(storage != NULL);
  storage->layout = layout;

  switch(layout)
  {
    case INTERLEAVED:
      storage->interleaved = rs_alloc_complex(size);
      break;
    case SPLIT:
      storage->split.real = rs_alloc_real(size);
      storage->split.imag = rs_alloc_real(size);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", layout);
      exit(EXIT_FAILURE);
  }
}

static void storage_free(storage_t *storage)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      rs_free(storage->interleaved);
      break;
    case SPLIT:
      rs_free(storage->split.real);
      rs_free(storage->split.imag);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

static void storage_set_elem(storage_t *storage, size_t offset, fftw_complex value)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      storage->interleaved[offset] = value;
      break;
    case SPLIT:
      storage->split.real[offset] = creal(value);
      storage->split.imag[offset] = cimag(value);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

static void execute_interpolate(interpolate_plan plan, storage_t *in, storage_t *out)
{
  assert(in != NULL);
  assert(out != NULL);
  assert(in->layout == out->layout);

  switch(in->layout)
  {
    case INTERLEAVED:
      interpolate_execute_interleaved(plan, in->interleaved, out->interleaved);
      break;
    case SPLIT:
      interpolate_execute_split(plan, in->split.real, in->split.imag, out->split.real, out->split.imag);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", in->layout);
      exit(EXIT_FAILURE);
  }
}

static fftw_complex storage_get_elem(const storage_t *storage, size_t offset)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      return storage->interleaved[offset];
    case SPLIT:
      return storage->split.real[offset] + storage->split.imag[offset] * I;
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

static fftw_complex test_function(double x, double y, double z)
{
  return ((1 + sin(2 * x)) * (3 + cos(3 * y)) * (5 + sin(1 * z))) +
         ((7 + cos(3 * x)) * (9 + sin(1 * y)) * (11 + cos(2 * z))) * I;
}

static void generate_test_data(storage_t *storage, const int z_width, const int y_width, const int x_width)
{
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

        storage_set_elem(storage, offset, test_function(x_pos, y_pos, z_pos));
      }
    }
  }
}

static double compute_delta_norm(const size_t size, const storage_t *in1, const storage_t *in2)
{
  double result_sq = 0.0;
  for(size_t offset = 0; offset < size; ++offset)
  {
    const fftw_complex val_1 = storage_get_elem(in1, offset);
    const fftw_complex val_2 = storage_get_elem(in2, offset);
    const double diff = cabs(val_1 - val_2);
    result_sq += diff * diff;
  }

  return sqrt(result_sq);
}

static void perform_timing(plan_constructor_t constructor,
  storage_layout_t layout, const int z_width, const int y_width, const int x_width)
{
  time_point_t begin_resample, end_resample, begin_plan, end_plan;

  time_point_save(&begin_plan);
  interpolate_plan plan = constructor(z_width, y_width, x_width, 0);
  time_point_save(&end_plan);

  const size_t block_size = z_width * y_width * x_width;

  storage_t coarse, fine, reference;
  storage_allocate(&coarse, layout, block_size);
  storage_allocate(&fine, layout, 8 * block_size);
  storage_allocate(&reference, layout, 8 * block_size);

  generate_test_data(&coarse, z_width, y_width, x_width);
  generate_test_data(&reference, z_width * 2, y_width * 2, x_width * 2);

  time_point_save(&begin_resample);
  execute_interpolate(plan, &coarse, &fine);
  time_point_save(&end_resample);

  const double abs_val = compute_delta_norm(8 * block_size, &fine, &reference);

  printf("Interpolation variant: %s\n", interpolate_get_name(plan));
  printf("Data layout: %s\n", layout_name(layout));
  printf("Problem size: %d x %d x %d\n", x_width, y_width, z_width);
  interpolate_print_timings(plan);
  printf("Planning time: %f\n", time_point_delta(&begin_plan, &end_plan));
  printf("Execution time: %f\n", time_point_delta(&begin_resample, &end_resample));
  printf("Delta: %f\n", abs_val);

  storage_free(&coarse);
  storage_free(&fine);
  storage_free(&reference);
  interpolate_destroy_plan(plan);
}

static void benchmark(FILE *file, storage_layout_t layout, plan_constructor_t *constructors)
{
  static const char* size_format_string = "%-8s";
  static const char* size_format_int = "%-8d";

  static const char* timing_format_string = "%-15s";
  static const char* timing_format_float = "%-15f";

  int plan_type_count = 0;

  while(constructors[plan_type_count] != NULL)
    ++plan_type_count;

  fprintf(file, "#data layout: %s\n", layout_name(layout));
  fprintf(file, size_format_string, "#size");
  for(int i=0; i < plan_type_count; ++i)
  {
    interpolate_plan plan = constructors[i](1, 1, 1, 0);
    fprintf(file, timing_format_string, interpolate_get_name(plan));
    interpolate_destroy_plan(plan);
  }

  fprintf(file, "\n");

  for(int size = 7; size < 100; size += 2)
  {
    const int runs = 10;

    size_t x_width, y_width, z_width;
    x_width = y_width = z_width = size;

    const size_t block_size = x_width * y_width * z_width;
    storage_t coarse, fine, reference;
    storage_allocate(&coarse, layout, block_size);
    storage_allocate(&fine, layout, 8 * block_size);
    storage_allocate(&reference, layout, 8 * block_size);

    generate_test_data(&coarse, z_width, y_width, x_width);
    generate_test_data(&reference, 2 * z_width, 2 * y_width, 2 * x_width);

    fprintf(file, size_format_int, size);

    for(int plan_type = 0; plan_type < plan_type_count; ++plan_type)
    {
      double time = 0.0;
      time_point_t begin_interpolate, end_interpolate;
      interpolate_plan plan = constructors[plan_type](x_width, y_width, z_width, 0);

      for(int run = 0; run < runs; ++run)
      {
        time_point_save(&begin_interpolate);
        execute_interpolate(plan, &coarse, &fine);
        time_point_save(&end_interpolate);
        time += time_point_delta(&begin_interpolate, &end_interpolate);
      }

      interpolate_destroy_plan(plan);
      time /= runs;
      fprintf(file, timing_format_float, time);

      assert(compute_delta_norm(8 * block_size, &fine, &reference) < 1e-5);
    }

    storage_free(&coarse);
    storage_free(&fine);
    storage_free(&reference);

    fprintf(file, "\n");
  }
}

int main(int argc, char **argv)
{
  if (argc > 1 && strcmp("--table-interleaved", argv[1]) == 0)
  {
    plan_constructor_t interleaved_constructors[] = {
      interpolate_plan_3d_naive_interleaved,
      interpolate_plan_3d_padding_aware_interleaved,
      interpolate_plan_3d_phase_shift_interleaved,
      interpolate_plan_3d_naive_cuda_interleaved,
      interpolate_plan_3d_padding_aware_cuda_interleaved,
      NULL
    };

    benchmark(stdout, INTERLEAVED, interleaved_constructors);
  }
  else if (argc > 1 && strcmp("--table-split", argv[1]) == 0)
  {
    plan_constructor_t split_constructors[] = {
      interpolate_plan_3d_naive_split,
      interpolate_plan_3d_padding_aware_split,
      interpolate_plan_3d_phase_shift_split,
      interpolate_plan_3d_naive_cuda_split,
      interpolate_plan_3d_padding_aware_cuda_split,
      NULL
    };

    benchmark(stdout, SPLIT, split_constructors);
  }
  else
  {
    int z = 75, y = 75, x = 75;

    perform_timing(interpolate_plan_3d_naive_interleaved, INTERLEAVED, z, y, x);
    printf("\n");
    perform_timing(interpolate_plan_3d_padding_aware_interleaved, INTERLEAVED, z, y, x);
    printf("\n");
    perform_timing(interpolate_plan_3d_phase_shift_interleaved, INTERLEAVED, z, y, x);

    printf("\n\n");
    perform_timing(interpolate_plan_3d_naive_split, SPLIT, z, y, x);
    printf("\n");
    perform_timing(interpolate_plan_3d_padding_aware_split, SPLIT, z, y, x);
    printf("\n");
    perform_timing(interpolate_plan_3d_phase_shift_split, SPLIT, z, y, x);
  }

  return EXIT_SUCCESS;
}
