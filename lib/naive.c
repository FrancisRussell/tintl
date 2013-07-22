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
  int interpolation;
  int dims[3];
  int strides[3];
  int fine_dims[3];
  int fine_strides[3];

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

static void plan_common(naive_plan plan, int n0, int n1, int n2, int flags);

static void pad_coarse_to_fine_interleaved(naive_plan plan, const fftw_complex *from, fftw_complex *to);
static void halve_nyquist_components(naive_plan plan, fftw_complex *coarse);

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

static void plan_common(naive_plan plan, int n0, int n1, int n2, int flags)
{
  plan->dims[0] = n2;
  plan->dims[1] = n1;
  plan->dims[2] = n0;

  for(int dim = 0; dim < 3; ++dim)
    plan->fine_dims[dim] = plan->dims[dim] * 2;

  plan->strides[0] = 1;
  plan->strides[1] = n2;
  plan->strides[2] = n2 * n1;

  plan->fine_strides[0] = 1;
  plan->fine_strides[1] = n2 * 2;
  plan->fine_strides[2] = n2 * n1 * 4;
}

interpolate_plan interpolate_plan_3d_naive_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  naive_plan plan = (naive_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, n0, n1, n2, flags);
  plan->interpolation = INTERLEAVED;

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  int rev_dims[] = { plan->dims[2], plan->dims[1], plan->dims[0] };
  int rev_fine_dims[] = { plan->fine_dims[2], plan->fine_dims[1], plan->fine_dims[0] };

  plan->naive_forward = fftw_plan_dft(3, rev_dims, scratch_coarse, scratch_coarse, FFTW_FORWARD, flags);
  plan->naive_backward = fftw_plan_dft(3, rev_fine_dims, scratch_fine, scratch_fine, FFTW_BACKWARD, flags);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);

  return wrapper;
}

interpolate_plan interpolate_plan_3d_naive_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  naive_plan plan = (naive_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, n0, n1, n2, flags);
  plan->interpolation = SPLIT;

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  double *const scratch_coarse_real = rs_alloc_real(block_size);
  double *const scratch_coarse_imag = rs_alloc_real(block_size);

  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  double *const scratch_fine_real = rs_alloc_real(8 * block_size);
  double *const scratch_fine_imag = rs_alloc_real(8 * block_size);

  fftw_iodim forward_dims[3];
  for(int dim = 0; dim < 3; ++dim)
  {
    forward_dims[2 - dim].n = plan->dims[dim];
    forward_dims[2 - dim].is = plan->strides[dim];
    forward_dims[2 - dim].os = plan->strides[dim] * 2;
  }

  plan->naive_forward = fftw_plan_guru_split_dft(3, forward_dims, 0, NULL,
      scratch_coarse_real, scratch_coarse_imag,
      (double*) scratch_fine, ((double*) scratch_fine)+1,
      flags);

  assert(plan->naive_forward != NULL);

  fftw_iodim backward_dims[3];
  for(int dim = 0; dim < 3; ++dim)
  {
    backward_dims[2 - dim].n = plan->dims[dim];
    backward_dims[2 - dim].is = plan->fine_strides[dim] * 2;
    backward_dims[2 - dim].os = plan->fine_strides[dim];
  }

  plan->naive_backward = fftw_plan_guru_split_dft(3, backward_dims, 0, NULL,
      ((double*) scratch_fine) + 1, (double*) scratch_fine,
      scratch_fine_imag, scratch_fine_real,
      flags);

  assert(plan->naive_backward != NULL);

  rs_free(scratch_fine_imag);
  rs_free(scratch_fine_real);

  rs_free(scratch_fine);

  rs_free(scratch_coarse_imag);
  rs_free(scratch_coarse_real);

  return wrapper;
}

interpolate_plan plan_interpolate_3d_naive_split_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = interpolate_plan_3d_naive_split(n0, n1, n2, flags);
  ((naive_plan) wrapper->detail)->interpolation = SPLIT_PRODUCT;
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
  assert(INTERLEAVED == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  fftw_complex *const input_copy = rs_alloc_complex(block_size);
  memcpy(input_copy, in, sizeof(fftw_complex) * block_size);

  time_point_save(&plan->before_forward);
  fftw_execute_dft(plan->naive_forward, input_copy, input_copy);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(plan, input_copy);
  pad_coarse_to_fine_interleaved(plan, input_copy, out);
  time_point_save(&plan->after_padding);
  fftw_execute_dft(plan->naive_backward, out, out);
  time_point_save(&plan->after_backward);

  rs_free(input_copy);
}

static void naive_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout)
{
  naive_plan plan = (naive_plan) detail;
  assert(SPLIT == plan->interpolation || SPLIT_PRODUCT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  time_point_save(&plan->before_forward);
  fftw_execute_split_dft(plan->naive_forward, rin, iin, (double*) scratch_coarse, ((double*) scratch_coarse) + 1);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(plan, scratch_coarse);
  pad_coarse_to_fine_interleaved(plan, scratch_coarse, scratch_fine);
  time_point_save(&plan->after_padding);
  fftw_execute_split_dft(plan->naive_backward, ((double*) scratch_fine) + 1, (double*) scratch_fine, iout, rout);
  time_point_save(&plan->after_backward);

  rs_free(scratch_fine);
  rs_free(scratch_coarse);
}

void naive_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out)
{
  naive_plan plan = (naive_plan) detail;
  assert(SPLIT_PRODUCT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  double *const scratch_fine = rs_alloc_real(8 * block_size);
  naive_interpolate_execute_split(detail, rin, iin, out, scratch_fine);
  pointwise_multiply_real(8 * block_size, out, scratch_fine);
}

void naive_interpolate_print_timings(const void *detail)
{
  naive_plan plan = (naive_plan) detail;
  printf("Forward: %f\n", time_point_delta(&plan->before_forward, &plan->after_forward));
  printf("Padding: %f\n", time_point_delta(&plan->after_forward, &plan->after_padding));
  printf("Backward: %f\n", time_point_delta(&plan->after_padding, &plan->after_backward));
}

static void block_copy_coarse_to_fine_interleaved(naive_plan plan, int n0, int n1, int n2, const fftw_complex *from, fftw_complex *to)
{
  assert(n0 <= plan->dims[0]);
  assert(n1 <= plan->dims[1]);
  assert(n2 <= plan->dims[2]);

  const double scale_factor = 1.0 / (plan->dims[0] * plan->dims[1] * plan->dims[2]);

  for(int i2=0; i2 < n2; ++i2)
  {
    for(int i1=0; i1 < n1; ++i1)
    {
      for(int i0=0; i0 < n0; ++i0)
        to[i0] = from[i0] * scale_factor;

      from += plan->strides[1];
      to += plan->fine_strides[1];
    }

    from += plan->strides[2] - n1 * plan->strides[1];
    to += plan->fine_strides[2] - n1 * plan->fine_strides[1];
  }
}

void halve_nyquist_components(naive_plan plan, fftw_complex *coarse)
{
  const int n2 = plan->dims[2];
  const int n1 = plan->dims[1];
  const int n0 = plan->dims[0];

  const int s2 = plan->strides[2];
  const int s1 = plan->strides[1];

  if (n2 % 2 == 0)
    for(int i1 = 0; i1 < n1; ++i1)
      for(int i0 = 0; i0 < n0; ++i0)
        coarse[s2 * (n2 / 2) +  s1 * i1 + i0] *= 0.5;

  if (n1 % 2 == 0)
    for(int i2 = 0; i2 < n2; ++i2)
      for(int i0 = 0; i0 < n0; ++i0)
        coarse[s2 * i2 +  s1 * (n1 / 2) + i0] *= 0.5;

  if (n0 % 2 == 0)
    for(int i2 = 0; i2 < n2; ++i2)
      for(int i1 = 0; i1 < n1; ++i1)
        coarse[s2 * i2 +  s1 * i1 + (n0 / 2)] *= 0.5;
}

static void pad_coarse_to_fine_interleaved(naive_plan plan, const fftw_complex *from, fftw_complex *to)
{
  const int coarse_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  memset(to, 0, 8 * coarse_size);

  int corner_flags[3];

  for(corner_flags[2] = 0; corner_flags[2] < 2; ++corner_flags[2])
  {
    for(corner_flags[1] = 0; corner_flags[1] < 2; ++corner_flags[1])
    {
      for(corner_flags[0] = 0; corner_flags[0] < 2; ++corner_flags[0])
      {
        const fftw_complex *coarse_block = from;
        fftw_complex *fine_block = to;
        int corner_sizes[3];

        for(int dim = 0; dim < 3; ++dim)
        {
          corner_sizes[dim] = corner_size(plan->dims[dim], corner_flags[dim]);
          const int coarse_index = (corner_flags[dim] == 0) ? 0 : plan->dims[dim] - corner_sizes[dim];
          const int fine_index = (corner_flags[dim] == 0) ? 0 : plan->fine_dims[dim] - corner_sizes[dim];

          coarse_block += plan->strides[dim] * coarse_index;
          fine_block += plan->fine_strides[dim] * fine_index;
        }

        block_copy_coarse_to_fine_interleaved(plan, corner_sizes[0], corner_sizes[1], corner_sizes[2], coarse_block, fine_block);
      }
    }
  }
}
