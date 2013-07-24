#include "interpolate.h"
#include "naive.h"
#include "timer.h"
#include "allocation.h"
#include "common.h"
#include "fftw_cycle.h"
#include <complex.h>
#include <stdint.h>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

typedef struct
{
  interpolate_properties_t props;

  fftw_plan naive_forward;
  fftw_plan naive_backward;

  time_point_t before_forward;
  time_point_t after_forward;
  time_point_t after_padding;
  time_point_t after_backward;
} naive_plan_s;

typedef naive_plan_s *naive_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */
static const char *get_name(const void *detail);
static void naive_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out);
static void naive_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout);
static void naive_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out);
static void naive_interpolate_print_timings(const void *detail);
static void naive_interpolate_destroy_detail(void *detail);

static void plan_common(naive_plan plan, interpolation_t type, int n0, int n1, int n2, int flags);

static const char *get_name(const void *detail)
{
  return "naive";
}

static interpolate_plan allocate_plan(void)
{
  interpolate_plan holder = malloc(sizeof(interpolate_plan_s));
  holder->detail = malloc(sizeof(naive_plan_s));
  holder->get_name = get_name;
  holder->execute_interleaved = naive_interpolate_execute_interleaved;
  holder->execute_split = naive_interpolate_execute_split;
  holder->execute_split_product = naive_interpolate_execute_split_product;
  holder->print_timings = naive_interpolate_print_timings;
  holder->destroy_detail = naive_interpolate_destroy_detail;

  return holder;
}

static void plan_common(naive_plan plan, interpolation_t type, int n0, int n1, int n2, int flags)
{
  populate_properties(&plan->props, type, n0, n1, n2);

  const size_t block_size = num_elements(&plan->props);

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  int rev_dims[] = { plan->props.dims[2], plan->props.dims[1], plan->props.dims[0] };
  int rev_fine_dims[] = { plan->props.fine_dims[2], plan->props.fine_dims[1], plan->props.fine_dims[0] };

  plan->naive_forward = fftw_plan_dft(3, rev_dims, scratch_coarse, scratch_coarse, FFTW_FORWARD, flags);
  plan->naive_backward = fftw_plan_dft(3, rev_fine_dims, scratch_fine, scratch_fine, FFTW_BACKWARD, flags);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

interpolate_plan interpolate_plan_3d_naive_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  naive_plan plan = (naive_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, INTERLEAVED, n0, n1, n2, flags);

  return wrapper;
}

interpolate_plan interpolate_plan_3d_naive_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  naive_plan plan = (naive_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, SPLIT, n0, n1, n2, flags);

  return wrapper;
}

interpolate_plan plan_interpolate_3d_naive_split_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = interpolate_plan_3d_naive_split(n0, n1, n2, flags);
  ((naive_plan) wrapper->detail)->props.type = SPLIT_PRODUCT;
  return wrapper;
}

static void naive_interpolate_destroy_detail(void *detail)
{
  naive_plan plan = (naive_plan) detail;

  fftw_destroy_plan(plan->naive_forward);
  fftw_destroy_plan(plan->naive_backward);

  free(plan);
}

static void naive_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out)
{
  naive_plan plan = (naive_plan) detail;
  assert(INTERLEAVED == plan->props.type);

  const size_t block_size = num_elements(&plan->props);

  fftw_complex *const input_copy = rs_alloc_complex(block_size);
  memcpy(input_copy, in, sizeof(fftw_complex) * block_size);

  time_point_save(&plan->before_forward);
  fftw_execute_dft(plan->naive_forward, input_copy, input_copy);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(&plan->props, input_copy);
  pad_coarse_to_fine_interleaved(&plan->props, input_copy, out);
  time_point_save(&plan->after_padding);
  fftw_execute_dft(plan->naive_backward, out, out);
  time_point_save(&plan->after_backward);

  rs_free(input_copy);
}

static void naive_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout)
{
  naive_plan plan = (naive_plan) detail;
  assert(SPLIT == plan->props.type || SPLIT_PRODUCT == plan->props.type);

  const size_t block_size = num_elements(&plan->props);

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  split_to_interleaved(block_size, rin, iin, scratch_coarse);
  time_point_save(&plan->before_forward);
  fftw_execute_dft(plan->naive_forward, scratch_coarse, scratch_coarse);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(&plan->props, scratch_coarse);
  pad_coarse_to_fine_interleaved(&plan->props, scratch_coarse, scratch_fine);
  time_point_save(&plan->after_padding);
  fftw_execute_dft(plan->naive_backward, scratch_fine, scratch_fine);
  time_point_save(&plan->after_backward);
  interleaved_to_split(8 * block_size, scratch_fine, rout, iout);

  rs_free(scratch_fine);
  rs_free(scratch_coarse);
}

void naive_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out)
{
  naive_plan plan = (naive_plan) detail;
  assert(SPLIT_PRODUCT == plan->props.type);

  const size_t block_size = num_elements(&plan->props);
  double *const scratch_fine = rs_alloc_real(8 * block_size);
  naive_interpolate_execute_split(detail, rin, iin, out, scratch_fine);
  pointwise_multiply_real(8 * block_size, out, scratch_fine);
  rs_free(scratch_fine);
}

void naive_interpolate_print_timings(const void *detail)
{
  naive_plan plan = (naive_plan) detail;
  printf("Forward: %f\n", time_point_delta(&plan->before_forward, &plan->after_forward));
  printf("Padding: %f\n", time_point_delta(&plan->after_forward, &plan->after_padding));
  printf("Backward: %f\n", time_point_delta(&plan->after_padding, &plan->after_backward));
}
