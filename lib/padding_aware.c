#include "interpolate.h"
#include "padding_aware.h"
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

typedef enum
{
  INTERLEAVED,
  SPLIT,
  SPLIT_PRODUCT
} interpolation_t;

typedef struct
{
  int interpolation;
  int dims[3];
  int strides[3];
  int fine_dims[3];
  int fine_strides[3];

  fftw_plan forward;
  fftw_plan n2_backward[2];
  fftw_plan n1_backward[2];
  fftw_plan n0_backward;

  time_point_t before_forward;
  time_point_t after_forward;
  time_point_t after_padding;
  time_point_t after_backward[3];
} pa_plan_s;

typedef pa_plan_s *pa_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */
static const char *get_name(const void *detail);
static void pa_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out);
static void pa_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout);
static void pa_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out);
static void pa_interpolate_print_timings(const void *detail);
static void pa_interpolate_destroy_detail(void *detail);

static void plan_common(pa_plan plan, int n0, int n1, int n2, int flags);

static void pad_coarse_to_fine_interleaved(pa_plan plan, const fftw_complex *from, fftw_complex *to);
static void halve_nyquist_components(pa_plan plan, fftw_complex *coarse);

static const char *get_name(const void *detail)
{
  return "padding-aware";
}

static interpolate_plan allocate_plan(void)
{
  interpolate_plan holder = malloc(sizeof(interpolate_plan_s));
  holder->detail = malloc(sizeof(pa_plan_s));
  holder->get_name = get_name;
  holder->execute_interleaved = pa_interpolate_execute_interleaved;
  holder->execute_split = pa_interpolate_execute_split;
  holder->execute_split_product = pa_interpolate_execute_split_product;
  holder->print_timings = pa_interpolate_print_timings;
  holder->destroy_detail = pa_interpolate_destroy_detail;

  return holder;
}

static int corner_size(const int n, const int negative)
{
  // In the even case, this will duplicate the Nyquist in both blocks
  return n / 2 + (negative == 0);
}

static void plan_common(pa_plan plan, int n0, int n1, int n2, int flags)
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

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  // Interpolation in direction 2, iteration in direction 0, positive frequencies
  plan->n2_backward[0] = fftw_plan_many_dft(1, &plan->fine_dims[2], corner_size(plan->dims[0], 0),
      scratch_fine, NULL, plan->fine_strides[2], plan->fine_strides[0],
      scratch_fine, NULL, plan->fine_strides[2], plan->fine_strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n2_backward[0] != NULL);

  // Interpolation in direction 2, iteration in direction 0, negative frequencies
  plan->n2_backward[1] = fftw_plan_many_dft(1, &plan->fine_dims[2], corner_size(plan->dims[0], 1),
      scratch_fine, NULL, plan->fine_strides[2], plan->fine_strides[0],
      scratch_fine, NULL, plan->fine_strides[2], plan->fine_strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n2_backward[1] != NULL);

  // Interpolation in direction 1, iteration in direction 0, positive frequencies
  plan->n1_backward[0] = fftw_plan_many_dft(1, &plan->fine_dims[1], corner_size(plan->dims[0], 0),
      scratch_fine, NULL, plan->fine_strides[1], plan->fine_strides[0],
      scratch_fine, NULL, plan->fine_strides[1], plan->fine_strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n1_backward[0] != NULL);

  // Interpolation in direction 1, iteration in direction 0, negative frequencies
  plan->n1_backward[1] = fftw_plan_many_dft(1, &plan->fine_dims[1], corner_size(plan->dims[0], 1),
      scratch_fine, NULL, plan->fine_strides[1], plan->fine_strides[0],
      scratch_fine, NULL, plan->fine_strides[1], plan->fine_strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n1_backward[1] != NULL);

  // Interpolation in direction 0, iteration in direction 1, all frequencies
  plan->n0_backward = fftw_plan_many_dft(1, &plan->fine_dims[0], plan->fine_dims[1] * plan->fine_dims[2],
      scratch_fine, NULL, plan->fine_strides[0], plan->fine_strides[1],
      scratch_fine, NULL, plan->fine_strides[0], plan->fine_strides[1],
      FFTW_BACKWARD, flags);
  assert(plan->n0_backward != NULL);

  rs_free(scratch_fine);
}

interpolate_plan interpolate_plan_3d_padding_aware_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  pa_plan plan = (pa_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, n0, n1, n2, flags);
  plan->interpolation = INTERLEAVED;

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);

  int rev_dims[] = { plan->dims[2], plan->dims[1], plan->dims[0] };
  plan->forward = fftw_plan_dft(3, rev_dims, scratch_coarse, scratch_coarse, FFTW_FORWARD, flags);

  rs_free(scratch_coarse);
  return wrapper;
}

interpolate_plan interpolate_plan_3d_padding_aware_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  pa_plan plan = (pa_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, n0, n1, n2, flags);
  plan->interpolation = INTERLEAVED;

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  double *const scratch_coarse_real = rs_alloc_real(block_size);
  double *const scratch_coarse_imag = rs_alloc_real(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  fftw_iodim forward_dims[3];
  for(int dim = 0; dim < 3; ++dim)
  {
    forward_dims[2 - dim].n = plan->dims[dim];
    forward_dims[2 - dim].is = plan->strides[dim];
    forward_dims[2 - dim].os = plan->strides[dim] * 2;
  }

  plan->forward = fftw_plan_guru_split_dft(3, forward_dims, 0, NULL,
      scratch_coarse_real, scratch_coarse_imag,
      (double*) scratch_fine, ((double*) scratch_fine)+1,
      flags);

  return wrapper;
}

interpolate_plan plan_interpolate_3d_padding_aware_split_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = interpolate_plan_3d_padding_aware_split(n0, n1, n2, flags);
  ((pa_plan) wrapper->detail)->interpolation = SPLIT_PRODUCT;
  return wrapper;
}

static void pa_interpolate_destroy_detail(void *detail)
{
  pa_plan plan = (pa_plan) detail;

  fftw_destroy_plan(plan->forward);

  fftw_destroy_plan(plan->n2_backward[0]);
  fftw_destroy_plan(plan->n2_backward[1]);
  fftw_destroy_plan(plan->n1_backward[0]);
  fftw_destroy_plan(plan->n1_backward[1]);
  fftw_destroy_plan(plan->n0_backward);

  free(plan);
}

static void backward_transform(const pa_plan plan, fftw_complex *data)
{
  int corner_sizes[3][2];

  for(int negative = 0; negative < 2; ++negative)
    for(int dim = 0; dim < 3; ++dim)
      corner_sizes[dim][negative] = corner_size(plan->dims[dim], negative);

  // Interpolation in direction 2
  for(int y = 0; y < corner_sizes[1][0]; ++y)
  {
    fftw_complex *positive_start = data + y * plan->fine_strides[1];
    fftw_execute_dft(plan->n2_backward[0], positive_start, positive_start);

    fftw_complex *negative_start = data + (y + 1) * plan->fine_strides[1] - corner_sizes[0][1];
    fftw_execute_dft(plan->n2_backward[1], negative_start, negative_start);
  }

  for(int y = plan->fine_dims[1] - corner_sizes[1][1]; y < plan->fine_dims[1]; ++y)
  {
    fftw_complex *positive_start = data + y * plan->fine_strides[1];
    fftw_execute_dft(plan->n2_backward[0], positive_start, positive_start);

    fftw_complex *negative_start = data + (y + 1) * plan->fine_strides[1] - corner_sizes[0][1];
    fftw_execute_dft(plan->n2_backward[1], negative_start, negative_start);
  }

  time_point_save(&plan->after_backward[2]);

  // Interpolation in direction 1
  for(int z = 0; z < plan->fine_dims[2]; ++z)
  {
    fftw_complex *positive_start = data + z * plan->fine_strides[2];
    fftw_execute_dft(plan->n1_backward[0], positive_start, positive_start);

    fftw_complex *negative_start = data + z * plan->fine_strides[2] + plan->fine_strides[1] - corner_sizes[0][1];
    fftw_execute_dft(plan->n1_backward[1], negative_start, negative_start);
  }

  time_point_save(&plan->after_backward[1]);

  // Interpolation in direction 0
  fftw_execute_dft(plan->n0_backward, data, data);

  time_point_save(&plan->after_backward[0]);
}

static void split_interleaved(const int size, fftw_complex *const in, double *rout, double *iout)
{
  for(int i=0; i<size; ++i)
  {
    rout[i] = creal(in[i]);
    iout[i] = cimag(in[i]);
  }
}

static void pa_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out)
{
  pa_plan plan = (pa_plan) detail;
  assert(INTERLEAVED == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  fftw_complex *const input_copy = rs_alloc_complex(block_size);
  memcpy(input_copy, in, sizeof(fftw_complex) * block_size);

  time_point_save(&plan->before_forward);
  fftw_execute_dft(plan->forward, input_copy, input_copy);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(plan, input_copy);
  pad_coarse_to_fine_interleaved(plan, input_copy, out);
  time_point_save(&plan->after_padding);
  backward_transform(plan, out);
  rs_free(input_copy);
}

static void pa_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout)
{
  pa_plan plan = (pa_plan) detail;
  assert(SPLIT == plan->interpolation || SPLIT_PRODUCT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  time_point_save(&plan->before_forward);
  fftw_execute_split_dft(plan->forward, rin, iin, (double*) scratch_coarse, ((double*) scratch_coarse) + 1);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(plan, scratch_coarse);
  pad_coarse_to_fine_interleaved(plan, scratch_coarse, scratch_fine);
  time_point_save(&plan->after_padding);
  backward_transform(plan, scratch_fine);
  split_interleaved(8 * block_size, scratch_fine, rout, iout);
  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

void pa_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out)
{
  pa_plan plan = (pa_plan) detail;
  assert(SPLIT_PRODUCT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  double *const scratch_fine = rs_alloc_real(8 * block_size);
  pa_interpolate_execute_split(detail, rin, iin, out, scratch_fine);
  pointwise_multiply_real(8 * block_size, out, scratch_fine);
}

void pa_interpolate_print_timings(const void *detail)
{
  pa_plan plan = (pa_plan) detail;
  printf("Forward: %f\n", time_point_delta(&plan->before_forward, &plan->after_forward));
  printf("Padding: %f\n", time_point_delta(&plan->after_forward, &plan->after_padding));
  for(int dim = 0; dim < 3; ++dim)
    printf("Backward %d: %f\n", dim, time_point_delta(&plan->after_padding, &plan->after_backward[dim]));
}

static void block_copy_coarse_to_fine_interleaved(pa_plan plan, int n0, int n1, int n2, const fftw_complex *from, fftw_complex *to)
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

void halve_nyquist_components(pa_plan plan, fftw_complex *coarse)
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

static void pad_coarse_to_fine_interleaved(pa_plan plan, const fftw_complex *from, fftw_complex *to)
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
