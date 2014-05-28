#include "tintl/timer.h"
#include "tintl/allocation.h"
#include "tintl/interpolate.h"
#include "tintl/padding_aware.h"
#include "tintl/phase_shift.h"
#include "tintl/naive.h"
#include "tintl/naive_cuda.h"
#include "tintl/padding_aware_cuda.h"
#include "storage.h"
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

static const double pi = 3.14159265358979323846;

typedef struct
{
  plan_constructor_t constructor;
  int statistic;
  int index;
  int field_width;
  int field_precision;
} column_info_t;

static column_info_t *construct_identical_column_info(plan_constructor_t *constructors, int statistic, int index,
  int field_width, int field_precision)
{
  int column_count = 0;
  while(constructors[column_count] != NULL)
    ++column_count;

  column_info_t *const columns = malloc(sizeof(column_info_t) * (column_count + 1));
  if (columns == NULL)
    return NULL;

  for(int i=0; i<column_count; ++i)
  {
    const column_info_t column_info =
    {
      .constructor = constructors[i],
      .statistic = statistic,
      .index = index,
      .field_width = field_width,
      .field_precision = field_precision
    };
    columns[i] = column_info;
  }

  columns[column_count].constructor = NULL;
  return columns;
}

static interpolate_plan interpolate_plan_3d_naive_split_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = interpolate_plan_3d_naive_split(n0, n1, n2, flags);

  if (plan == NULL)
    return NULL;

  interpolate_set_flags(plan, PREFER_SPLIT_LAYOUT);
  return plan;
}

static interpolate_plan interpolate_plan_3d_naive_split_packed(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = interpolate_plan_3d_naive_split(n0, n1, n2, flags);

  if (plan == NULL)
    return NULL;

  interpolate_set_flags(plan, PREFER_PACKED_LAYOUT);
  return plan;
}

static interpolate_plan interpolate_plan_3d_padding_aware_split_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = interpolate_plan_3d_padding_aware_split(n0, n1, n2, flags);

  if (plan == NULL)
    return NULL;

  interpolate_set_flags(plan, PREFER_SPLIT_LAYOUT);
  return plan;
}

static interpolate_plan interpolate_plan_3d_padding_aware_split_packed(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = interpolate_plan_3d_padding_aware_split(n0, n1, n2, flags);

  if (plan == NULL)
    return NULL;

  interpolate_set_flags(plan, PREFER_PACKED_LAYOUT);
  return plan;
}

static interpolate_plan interpolate_plan_3d_phase_shift_split_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = interpolate_plan_3d_phase_shift_split(n0, n1, n2, flags);

  if (plan == NULL)
    return NULL;

  interpolate_set_flags(plan, PREFER_SPLIT_LAYOUT);
  return plan;
}

static interpolate_plan interpolate_plan_3d_phase_shift_split_packed(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = interpolate_plan_3d_phase_shift_split(n0, n1, n2, flags);

  if (plan == NULL)
    return NULL;

  interpolate_set_flags(plan, PREFER_PACKED_LAYOUT);
  return plan;
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

static void benchmark(FILE *file, storage_layout_t layout, column_info_t *cols)
{
  static const char* size_format_string = "%-8s";
  static const char* size_format_int = "%-8d";

  int col_count = 0;

  while(cols[col_count].constructor != NULL)
    ++col_count;

  int *const use_cols = malloc(sizeof(int) * col_count);
  assert(use_cols != NULL);

  fprintf(file, "#data layout: %s\n", layout_name(layout));
  fprintf(file, size_format_string, "#size");

  stat_type_t stat_type = STATISTIC_UNKNOWN;
  for(int i=0; i < col_count; ++i)
  {
    // We currently have to construct a plan to get a column name. This is
    // problematic if we have no idea which sizes the constructor will return
    // non-NULL plans for.
    interpolate_plan plan = cols[i].constructor(7, 7, 7, 0);

    if (plan != NULL)
    {
      fprintf(file, "%-*s", cols[i].field_width, interpolate_get_name(plan));
      double dummy;
      interpolate_get_statistic_float(plan, cols[i].statistic, cols[i].index, &stat_type, &dummy);
      interpolate_destroy_plan(plan);
      use_cols[i] = 1;
    }
    else
    {
      use_cols[i] = 0;
    }
  }

  fprintf(file, "\n");

  for(int size = 7; size <= 120; ++size)
  {
    const int runs = 10;

    size_t x_width, y_width, z_width;
    x_width = y_width = z_width = size;

    const size_t block_size = x_width * y_width * z_width;
    fprintf(file, size_format_int, size);

    for(int col = 0; col < col_count; ++col)
    {
      if (use_cols[col] == 0)
        continue;

      int missing_measurement = 0;
      double measurement = 0.0;

      if (stat_type == STATISTIC_PLANNING)
      {
        for(int run = 0; run < runs; ++run)
        {
          interpolate_plan plan = cols[col].constructor(x_width, y_width, z_width, 0);

          if (plan == NULL)
          {
            missing_measurement = 1;
          }
          else
          {
            double run_measurement;
            interpolate_get_statistic_float(plan, cols[col].statistic, cols[col].index, &stat_type, &run_measurement);
            measurement += run_measurement;
            interpolate_destroy_plan(plan);
          }
        }
      }
      else
      {
        storage_t coarse, fine, reference;
        storage_allocate(&coarse, layout, block_size);
        storage_allocate(&fine, layout, 8 * block_size);
        storage_allocate(&reference, layout, 8 * block_size);

        generate_test_data(&coarse, z_width, y_width, x_width);
        generate_test_data(&reference, 2 * z_width, 2 * y_width, 2 * x_width);

        interpolate_plan plan = cols[col].constructor(x_width, y_width, z_width, 0);

        if (plan == NULL)
        {
          missing_measurement = 1;
        }
        else
        {
          storage_zero(&fine);

          for(int run = 0; run < runs; ++run)
          {
            execute_interpolate(plan, &coarse, &fine);
            double run_measurement;
            interpolate_get_statistic_float(plan, cols[col].statistic, cols[col].index, &stat_type, &run_measurement);
            measurement += run_measurement;
          }

          assert(compute_delta_norm(8 * block_size, &fine, &reference) < 1e-5);
          interpolate_destroy_plan(plan);
        }

        storage_free(&coarse);
        storage_free(&fine);
        storage_free(&reference);
      }

      measurement /= runs;

      if (missing_measurement)
        fprintf(file, "%-*s", cols[col].field_width, "-");
      else
        fprintf(file, "%-*.*f", cols[col].field_width, cols[col].field_precision, measurement);
    }

    fprintf(file, "\n");
  }

  free(use_cols);
}

int main(int argc, char **argv)
{
  const int seconds_width=15;
  const int seconds_precision=6;

  const int ticks_width=15;
  const int ticks_precision=0;

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

    column_info_t *const cols = construct_identical_column_info(interleaved_constructors, STATISTIC_EXECUTION_TIME, 0,
      seconds_width, seconds_precision);
    assert(cols != NULL);
    benchmark(stdout, INTERLEAVED, cols);
    free(cols);
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

    column_info_t *const cols = construct_identical_column_info(split_constructors, STATISTIC_EXECUTION_TIME, 0,
      seconds_width, seconds_precision);
    assert(cols != NULL);
    benchmark(stdout, SPLIT, cols);
    free(cols);
  }
  else if (argc > 1 && strcmp("--table-layouts", argv[1]) == 0)
  {
    plan_constructor_t split_constructors[] = {
      interpolate_plan_3d_naive_split_split,
      interpolate_plan_3d_naive_split_packed,
      interpolate_plan_3d_padding_aware_split_split,
      interpolate_plan_3d_padding_aware_split_packed,
      interpolate_plan_3d_phase_shift_split_split,
      interpolate_plan_3d_phase_shift_split_packed,
      NULL
    };

    fprintf(stdout, "#format order: split, packed\n");
    column_info_t *const cols = construct_identical_column_info(split_constructors, STATISTIC_EXECUTION_TIME, 0,
      seconds_width, seconds_precision);
    assert(cols != NULL);
    benchmark(stdout, SPLIT, cols);
    free(cols);
  }
  else if (argc > 1 && strcmp("--table-phase-shift-batching", argv[1]) == 0)
  {
    column_info_t cols[] = {
      { .constructor = interpolate_plan_3d_phase_shift_interleaved, .field_width = ticks_width,
        .field_precision = ticks_precision, .statistic = PHASE_SHIFT_STATISTIC_BATCH_TRANSFORMS, .index = 0 },
      { .constructor = interpolate_plan_3d_phase_shift_interleaved, .field_width = ticks_width,
        .field_precision = ticks_precision, .statistic = PHASE_SHIFT_STATISTIC_INDIVIDUAL_TRANSFORMS, .index = 0 },
      { .constructor = interpolate_plan_3d_phase_shift_interleaved, .field_width = ticks_width,
        .field_precision = ticks_precision, .statistic = PHASE_SHIFT_STATISTIC_BATCH_TRANSFORMS, .index = 1 },
      { .constructor = interpolate_plan_3d_phase_shift_interleaved, .field_width = ticks_width,
        .field_precision = ticks_precision, .statistic = PHASE_SHIFT_STATISTIC_INDIVIDUAL_TRANSFORMS, .index = 1 },
      { .constructor = interpolate_plan_3d_phase_shift_interleaved, .field_width = ticks_width,
        .field_precision = ticks_precision, .statistic = PHASE_SHIFT_STATISTIC_BATCH_TRANSFORMS, .index = 2 },
      { .constructor = interpolate_plan_3d_phase_shift_interleaved, .field_width = ticks_width,
        .field_precision = ticks_precision, .statistic = PHASE_SHIFT_STATISTIC_INDIVIDUAL_TRANSFORMS, .index = 2 },
      { .constructor = NULL }
    };

    fprintf(stdout, "#batch(dim=0), individual(dim=0), batch(dim=1), individual(dim=1), batch(dim=2), individual(dim=2)\n");
    benchmark(stdout, INTERLEAVED, cols);
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
