#include "interpolate.h"
#include "naive.h"
#include "timer.h"
#include "allocation.h"
#include "common.h"
#include "fftw_utility.h"
#include "fftw_cycle.h"
#include <complex.h>
#include <stdint.h>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

typedef enum
{
  PACKED,
  SEPARATE
} strategy_t;

/// Implementation-specific structure for naive interpolation plans.
typedef struct
{
  struct interpolate_plan_s common;
  strategy_t strategy;

  fftw_plan interleaved_forward;
  fftw_plan interleaved_backward;

  fftw_plan real_forward;
  fftw_plan real_backward;

  time_point_t before_forward;
  time_point_t after_forward;
  time_point_t after_padding;
  time_point_t after_backward;
} naive_plan_s;

typedef naive_plan_s *naive_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */
static const char *get_name(const interpolate_plan plan);
static void naive_set_flags(interpolate_plan plan, int flags);
static void naive_get_statistic_float(const interpolate_plan plan, int statistic, int index, stat_type_t *type, double *result);
static void naive_interpolate_execute_interleaved(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void naive_interpolate_execute_split(interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);
static void naive_interpolate_execute_split_product(interpolate_plan plan, double *rin, double *iin, double *out);
static void naive_interpolate_print_timings(const interpolate_plan plan);
static void naive_interpolate_destroy_detail(interpolate_plan plan);

static void plan_common(naive_plan plan, interpolation_t type, int n0, int n1, int n2, int flags);

static const char *get_name(const interpolate_plan plan)
{
  return "naive";
}

static interpolate_plan allocate_plan(void)
{
  setup_threading();

  interpolate_plan holder = malloc(sizeof(naive_plan_s));
  assert(holder != NULL);

  holder->ref_cnt = 1;

  holder->get_name = get_name;
  holder->set_flags = naive_set_flags;
  holder->get_statistic_float = naive_get_statistic_float;
  holder->execute_interleaved = naive_interpolate_execute_interleaved;
  holder->execute_split = naive_interpolate_execute_split;
  holder->execute_split_product = naive_interpolate_execute_split_product;
  holder->print_timings = naive_interpolate_print_timings;
  holder->destroy_detail = naive_interpolate_destroy_detail;

  return holder;
}

static void plan_common(naive_plan plan, interpolation_t type, int n0, int n1, int n2, int flags)
{
  populate_properties((interpolate_plan) plan, type, n0, n1, n2);

  interpolate_plan parent = cast_to_plan(plan);
  const size_t block_size = num_elements(parent);

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  block_info_t coarse_info, fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);

  int rev_dims[] = { coarse_info.dims[2], coarse_info.dims[1], coarse_info.dims[0] };
  int rev_fine_dims[] = { fine_info.dims[2], fine_info.dims[1], fine_info.dims[0] };

  plan->real_forward = NULL;
  plan->real_backward = NULL;

  plan->interleaved_forward = fftw_plan_dft(3, rev_dims, scratch_coarse, scratch_coarse, FFTW_FORWARD, flags);
  plan->interleaved_backward = fftw_plan_dft(3, rev_fine_dims, scratch_fine, scratch_fine, FFTW_BACKWARD, flags);

  assert(plan->interleaved_forward != NULL);
  assert(plan->interleaved_backward != NULL);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

static void naive_set_flags(interpolate_plan parent, const int flags)
{
  naive_plan plan = (naive_plan) parent;

  const int conflicting_layouts = PREFER_PACKED_LAYOUT | PREFER_SPLIT_LAYOUT;
  assert((flags & conflicting_layouts) != conflicting_layouts);

  if (flags & PREFER_PACKED_LAYOUT)
    plan->strategy = PACKED;

  if (flags & PREFER_SPLIT_LAYOUT)
    plan->strategy = SEPARATE;
}

static void naive_get_statistic_float(const interpolate_plan parent, int statistic, int index, stat_type_t *type, double *result)
{
  *type = STATISTIC_UNKNOWN;
}

interpolate_plan interpolate_plan_3d_naive_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan parent = allocate_plan();
  naive_plan plan = (naive_plan) parent;

  flags |= FFTW_MEASURE;
  plan_common(plan, INTERPOLATE_INTERLEAVED, n0, n1, n2, flags);
  plan->strategy = PACKED;

  return parent;
}

interpolate_plan interpolate_plan_3d_naive_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan parent = allocate_plan();
  naive_plan plan = (naive_plan) parent;

  flags |= FFTW_MEASURE;
  plan_common(plan, INTERPOLATE_SPLIT, n0, n1, n2, flags);

  block_info_t coarse_info, fine_info, transformed_coarse_info, transformed_fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);
  get_block_info_real_recip_coarse(parent, &transformed_coarse_info);
  get_block_info_real_recip_fine(parent, &transformed_fine_info);

  const size_t block_size = num_elements_block(&coarse_info);
  const size_t transformed_size_coarse = num_elements_block(&transformed_coarse_info);
  const size_t transformed_size_fine = num_elements_block(&transformed_fine_info);

  int rev_dims[] = { coarse_info.dims[2], coarse_info.dims[1], coarse_info.dims[0] };
  int rev_fine_dims[] = { fine_info.dims[2], fine_info.dims[1], fine_info.dims[0] };

  double *const scratch_coarse_real = rs_alloc_real(block_size);
  fftw_complex *const scratch_coarse_complex = rs_alloc_complex(transformed_size_coarse);

  double *const scratch_fine_real = rs_alloc_real(8 * block_size);
  fftw_complex *const scratch_fine_complex = rs_alloc_complex(transformed_size_fine);

  plan->real_forward = fftw_plan_dft_r2c(3, rev_dims, scratch_coarse_real, scratch_coarse_complex, flags);
  plan->real_backward = fftw_plan_dft_c2r(3, rev_fine_dims, scratch_fine_complex, scratch_fine_real, flags);

  assert(plan->real_forward != NULL);
  assert(plan->real_backward != NULL);

  rs_free(scratch_coarse_real);
  rs_free(scratch_coarse_complex);
  rs_free(scratch_fine_real);
  rs_free(scratch_fine_complex);

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split(parent);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split(parent);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;

  return parent;
}

interpolate_plan interpolate_plan_3d_naive_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan parent = interpolate_plan_3d_naive_split(n0, n1, n2, flags);
  parent->type = INTERPOLATE_SPLIT_PRODUCT;
  naive_plan plan = (naive_plan) parent;

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split_product(parent);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split_product(parent);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;
  return parent;
}

static void naive_interpolate_destroy_detail(interpolate_plan parent)
{
  naive_plan plan = (naive_plan) parent;

  fftw_destroy_plan(plan->interleaved_forward);
  fftw_destroy_plan(plan->interleaved_backward);

  fftw_destroy_plan_maybe_null(plan->real_forward);
  fftw_destroy_plan_maybe_null(plan->real_backward);
}

static void naive_interpolate_execute_interleaved(interpolate_plan parent, fftw_complex *in, fftw_complex *out)
{
  naive_plan plan = (naive_plan) parent;
  assert(plan->strategy == PACKED);

  block_info_t coarse_info, fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);

  const size_t block_size = num_elements_block(&coarse_info);
  fftw_complex *const input_copy = rs_alloc_complex(block_size);

  memcpy(input_copy, in, sizeof(fftw_complex) * block_size);
  time_point_save(&plan->before_forward);
  fftw_execute_dft(plan->interleaved_forward, input_copy, input_copy);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(parent, &coarse_info, input_copy);
  pad_coarse_to_fine_interleaved(parent, &coarse_info, input_copy, &fine_info, out, 0);
  time_point_save(&plan->after_padding);
  fftw_execute_dft(plan->interleaved_backward, out, out);
  time_point_save(&plan->after_backward);

  rs_free(input_copy);
}

static void naive_interpolate_real(naive_plan plan, double *in, double *out)
{
  interpolate_plan parent = cast_to_plan(plan);

  block_info_t coarse_info, transformed_coarse_info, transformed_fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_real_recip_coarse(parent, &transformed_coarse_info);
  get_block_info_real_recip_fine(parent, &transformed_fine_info);

  const size_t transformed_size_coarse = num_elements_block(&transformed_coarse_info);
  const size_t transformed_size_fine = num_elements_block(&transformed_fine_info);

  fftw_complex *const scratch_coarse = rs_alloc_complex(transformed_size_coarse);
  fftw_complex *const scratch_fine = rs_alloc_complex(transformed_size_fine);

  time_point_save(&plan->before_forward);
  fftw_execute_dft_r2c(plan->real_forward, in, scratch_coarse);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(parent, &transformed_coarse_info, scratch_coarse);
  pad_coarse_to_fine_interleaved(parent, &transformed_coarse_info, scratch_coarse, &transformed_fine_info, scratch_fine, 1);
  time_point_save(&plan->after_padding);
  fftw_execute_dft_c2r(plan->real_backward, scratch_fine, out);
  time_point_save(&plan->after_backward);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

static void naive_interpolate_execute_split(interpolate_plan parent, double *rin, double *iin, double *rout, double *iout)
{
  naive_plan plan = (naive_plan) parent;
  assert(INTERPOLATE_SPLIT == parent->type || INTERPOLATE_SPLIT_PRODUCT == parent->type);

  if (plan->strategy == PACKED)
  {
    block_info_t coarse_info, fine_info;
    get_block_info_coarse(parent, &coarse_info);
    get_block_info_fine(parent, &fine_info);
    const size_t block_size = num_elements_block(&coarse_info);

    fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
    fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

    interleave_real(block_size, (double*) scratch_coarse, rin, iin);
    time_point_save(&plan->before_forward);
    fftw_execute_dft(plan->interleaved_forward, scratch_coarse, scratch_coarse);
    time_point_save(&plan->after_forward);
    halve_nyquist_components(parent, &coarse_info, scratch_coarse);
    pad_coarse_to_fine_interleaved(parent, &coarse_info, scratch_coarse, &fine_info, scratch_fine, 0);
    time_point_save(&plan->after_padding);
    fftw_execute_dft(plan->interleaved_backward, scratch_fine, scratch_fine);
    time_point_save(&plan->after_backward);
    deinterleave_real(8 * block_size, (const double*) scratch_fine, rout, iout);

    rs_free(scratch_fine);
    rs_free(scratch_coarse);
  }
  else if (plan->strategy == SEPARATE)
  {
    naive_interpolate_real(plan, rin, rout);
    naive_interpolate_real(plan, iin, iout);
  }
  else
  {
    assert(0 && "Unknown strategy.");
  }
}

void naive_interpolate_execute_split_product(interpolate_plan parent, double *rin, double *iin, double *out)
{
  naive_plan plan = (naive_plan) parent;
  assert(INTERPOLATE_SPLIT_PRODUCT == parent->type);

  const size_t block_size = num_elements(parent);

  if (plan->strategy == PACKED)
  {
    fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
    fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

    interleave_real(block_size, (double*) scratch_coarse, rin, iin);
    naive_interpolate_execute_interleaved(parent, scratch_coarse, scratch_fine);
    complex_to_product(8 * block_size, scratch_fine, out);

    rs_free(scratch_coarse);
    rs_free(scratch_fine);
  }
  else if (plan->strategy == SEPARATE)
  {
    double *const scratch_fine = rs_alloc_real(8 * block_size);
    naive_interpolate_execute_split(parent, rin, iin, out, scratch_fine);
    pointwise_multiply_real(8 * block_size, out, scratch_fine);
    rs_free(scratch_fine);
  }
  else
  {
    assert(0 && "Unknown strategy");
  }
}

void naive_interpolate_print_timings(const interpolate_plan parent)
{
  naive_plan plan = (naive_plan) parent;
  printf("Forward: %f\n", time_point_delta(&plan->before_forward, &plan->after_forward));
  printf("Padding: %f\n", time_point_delta(&plan->after_forward, &plan->after_padding));
  printf("Backward: %f\n", time_point_delta(&plan->after_padding, &plan->after_backward));
}
