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
  PACKED,
  SEPARATE
} strategy_t;

typedef struct
{
  interpolate_properties_t props;
  strategy_t strategy;

  fftw_plan interleaved_forward;
  fftw_plan n2_backward_interleaved[2];
  fftw_plan n1_backward_interleaved[2];
  fftw_plan n0_backward_interleaved;

  fftw_plan real_forward;
  fftw_plan n2_backward_real;
  fftw_plan n1_backward_real;
  fftw_plan n0_backward_real;

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

static void plan_common(pa_plan plan, interpolation_t type, int n0, int n1, int n2, int flags);

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

static void plan_common(pa_plan plan, interpolation_t type, int n0, int n1, int n2, int flags)
{
  populate_properties(&plan->props, type, n0, n1, n2);
  const size_t block_size = num_elements(&plan->props);

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  block_info_t fine_info;
  get_block_info_fine(&plan->props, &fine_info);

  int rev_dims[] = { plan->props.dims[2], plan->props.dims[1], plan->props.dims[0] };
  plan->interleaved_forward = fftw_plan_dft(3, rev_dims, scratch_coarse, scratch_coarse, FFTW_FORWARD, flags);
  plan->real_forward = NULL;

  plan->n2_backward_real = NULL;
  plan->n1_backward_real = NULL;
  plan->n0_backward_real = NULL;

  // Interpolation in direction 2, iteration in direction 0, positive frequencies
  plan->n2_backward_interleaved[0] = fftw_plan_many_dft(1, &fine_info.dims[2], corner_size(plan->props.dims[0], 0),
      scratch_fine, NULL, fine_info.strides[2], fine_info.strides[0],
      scratch_fine, NULL, fine_info.strides[2], fine_info.strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n2_backward_interleaved[0] != NULL);

  // Interpolation in direction 2, iteration in direction 0, negative frequencies
  plan->n2_backward_interleaved[1] = fftw_plan_many_dft(1, &fine_info.dims[2], corner_size(plan->props.dims[0], 1),
      scratch_fine, NULL, fine_info.strides[2], fine_info.strides[0],
      scratch_fine, NULL, fine_info.strides[2], fine_info.strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n2_backward_interleaved[1] != NULL);

  // Interpolation in direction 1, iteration in direction 0, positive frequencies
  plan->n1_backward_interleaved[0] = fftw_plan_many_dft(1, &fine_info.dims[1], corner_size(plan->props.dims[0], 0),
      scratch_fine, NULL, fine_info.strides[1], fine_info.strides[0],
      scratch_fine, NULL, fine_info.strides[1], fine_info.strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n1_backward_interleaved[0] != NULL);

  // Interpolation in direction 1, iteration in direction 0, negative frequencies
  plan->n1_backward_interleaved[1] = fftw_plan_many_dft(1, &fine_info.dims[1], corner_size(plan->props.dims[0], 1),
      scratch_fine, NULL, fine_info.strides[1], fine_info.strides[0],
      scratch_fine, NULL, fine_info.strides[1], fine_info.strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n1_backward_interleaved[1] != NULL);

  // Interpolation in direction 0, iteration in direction 1, all frequencies
  plan->n0_backward_interleaved = fftw_plan_many_dft(1, &fine_info.dims[0], fine_info.dims[1] * fine_info.dims[2],
      scratch_fine, NULL, fine_info.strides[0], fine_info.strides[1],
      scratch_fine, NULL, fine_info.strides[0], fine_info.strides[1],
      FFTW_BACKWARD, flags);
  assert(plan->n0_backward_interleaved != NULL);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

interpolate_plan interpolate_plan_3d_padding_aware_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  pa_plan plan = (pa_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, INTERLEAVED, n0, n1, n2, flags);
  plan->strategy = PACKED;

  return wrapper;
}

interpolate_plan interpolate_plan_3d_padding_aware_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  pa_plan plan = (pa_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, SPLIT, n0, n1, n2, flags);

  block_info_t coarse_info, fine_info, transformed_coarse_info, transformed_fine_info;
  get_block_info_coarse(&plan->props, &coarse_info);
  get_block_info_fine(&plan->props, &fine_info);
  get_block_info_real_recip_coarse(&plan->props, &transformed_coarse_info);
  get_block_info_real_recip_fine(&plan->props, &transformed_fine_info);

  const size_t block_size = num_elements_block(&coarse_info);
  const size_t transformed_size_coarse = num_elements_block(&transformed_coarse_info);
  const size_t transformed_size_fine = num_elements_block(&transformed_fine_info);

  double *const scratch_coarse_real = rs_alloc_real(block_size);
  fftw_complex *const scratch_coarse_complex = rs_alloc_complex(transformed_size_coarse);

  fftw_complex *const scratch_fine_complex = rs_alloc_complex(transformed_size_fine);
  double *const scratch_fine_real = rs_alloc_real(8 * block_size);

  int rev_dims[] = { plan->props.dims[2], plan->props.dims[1], plan->props.dims[0] };
  plan->real_forward = fftw_plan_dft_r2c(3, rev_dims, scratch_coarse_real, scratch_coarse_complex, flags);

  // Interpolation in direction 2, iteration in direction 0, positive frequencies
  plan->n2_backward_real = fftw_plan_many_dft(1, &transformed_fine_info.dims[2], corner_size(plan->props.dims[0], 0),
      scratch_fine_complex, NULL, transformed_fine_info.strides[2], transformed_fine_info.strides[0],
      scratch_fine_complex, NULL, transformed_fine_info.strides[2], transformed_fine_info.strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n2_backward_real != NULL);

  // Interpolation in direction 1, iteration in direction 0, positive frequencies
  plan->n1_backward_real = fftw_plan_many_dft(1, &transformed_fine_info.dims[1], corner_size(plan->props.dims[0], 0),
      scratch_fine_complex, NULL, transformed_fine_info.strides[1], transformed_fine_info.strides[0],
      scratch_fine_complex, NULL, transformed_fine_info.strides[1], transformed_fine_info.strides[0],
      FFTW_BACKWARD, flags);
  assert(plan->n1_backward_real != NULL);

  // Interpolation in direction 0, iteration in direction 1, all frequencies
  plan->n0_backward_real = fftw_plan_many_dft_c2r(1, &fine_info.dims[0], fine_info.dims[1] * fine_info.dims[2],
      scratch_fine_complex, NULL, transformed_fine_info.strides[0], transformed_fine_info.strides[1],
      scratch_fine_real,    NULL, fine_info.strides[0],             fine_info.strides[1],
      flags);
  assert(plan->n0_backward_real != NULL);

  rs_free(scratch_coarse_real);
  rs_free(scratch_coarse_complex);
  rs_free(scratch_fine_real);
  rs_free(scratch_fine_complex);

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split(wrapper, &plan->props);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split(wrapper, &plan->props);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;

  return wrapper;
}

interpolate_plan plan_interpolate_3d_padding_aware_split_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = interpolate_plan_3d_padding_aware_split(n0, n1, n2, flags);
  pa_plan plan = (pa_plan) wrapper->detail;
  plan->props.type = SPLIT_PRODUCT;

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split_product(wrapper, &plan->props);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split_product(wrapper, &plan->props);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;

  return wrapper;
}

static void pa_interpolate_destroy_detail(void *detail)
{
  pa_plan plan = (pa_plan) detail;

  fftw_destroy_plan(plan->interleaved_forward);

  for(int corner = 0; corner < 2; ++corner)
  {
    fftw_destroy_plan(plan->n2_backward_interleaved[corner]);
    fftw_destroy_plan(plan->n1_backward_interleaved[corner]);
  }

  fftw_destroy_plan(plan->n0_backward_interleaved);

  if (plan->real_forward != NULL)
    fftw_destroy_plan(plan->real_forward);

  if (plan->n2_backward_real != NULL)
    fftw_destroy_plan(plan->n2_backward_real);

  if (plan->n1_backward_real != NULL)
    fftw_destroy_plan(plan->n1_backward_real);

  if (plan->n0_backward_real != NULL)
    fftw_destroy_plan(plan->n0_backward_real);

  free(plan);
}

static void backward_transform_c2c(const pa_plan plan, const block_info_t *data_info, fftw_complex *data)
{
  size_t corner_sizes[3][2];

  for(int negative = 0; negative < 2; ++negative)
    for(int dim = 0; dim < 3; ++dim)
      corner_sizes[dim][negative] = corner_size(plan->props.dims[dim], negative);

  // Interpolation in direction 2
  for(size_t y = 0; y < corner_sizes[1][0]; ++y)
  {
    fftw_complex *positive_start = data + y * data_info->strides[1];
    fftw_execute_dft(plan->n2_backward_interleaved[0], positive_start, positive_start);

    fftw_complex *negative_start = data + (y + 1) * data_info->strides[1] - corner_sizes[0][1];
    fftw_execute_dft(plan->n2_backward_interleaved[1], negative_start, negative_start);
  }

  for(size_t y = data_info->dims[1] - corner_sizes[1][1]; y < data_info->dims[1]; ++y)
  {
    fftw_complex *positive_start = data + y * data_info->strides[1];
    fftw_execute_dft(plan->n2_backward_interleaved[0], positive_start, positive_start);

    fftw_complex *negative_start = data + (y + 1) * data_info->strides[1] - corner_sizes[0][1];
    fftw_execute_dft(plan->n2_backward_interleaved[1], negative_start, negative_start);
  }

  time_point_save(&plan->after_backward[2]);

  // Interpolation in direction 1
  for(size_t z = 0; z < data_info->dims[2]; ++z)
  {
    fftw_complex *positive_start = data + z * data_info->strides[2];
    fftw_execute_dft(plan->n1_backward_interleaved[0], positive_start, positive_start);

    fftw_complex *negative_start = data + z * data_info->strides[2] + data_info->strides[1] - corner_sizes[0][1];
    fftw_execute_dft(plan->n1_backward_interleaved[1], negative_start, negative_start);
  }

  time_point_save(&plan->after_backward[1]);

  // Interpolation in direction 0
  fftw_execute_dft(plan->n0_backward_interleaved, data, data);

  time_point_save(&plan->after_backward[0]);
}

static void backward_transform_c2r(const pa_plan plan,
  const block_info_t *from_info, fftw_complex *from,
  const block_info_t *to_info, double *to)
{
  size_t corner_sizes[3][2];

  for(int negative = 0; negative < 2; ++negative)
    for(int dim = 0; dim < 3; ++dim)
      corner_sizes[dim][negative] = corner_size(plan->props.dims[dim], negative);

  // Interpolation in direction 2
  for(size_t y = 0; y < corner_sizes[1][0]; ++y)
  {
    fftw_complex *positive_start = from + y * from_info->strides[1];
    fftw_execute_dft(plan->n2_backward_real, positive_start, positive_start);
  }

  for(size_t y = to_info->dims[1] - corner_sizes[1][1]; y < to_info->dims[1]; ++y)
  {
    fftw_complex *positive_start = from + y * from_info->strides[1];
    fftw_execute_dft(plan->n2_backward_real, positive_start, positive_start);
  }

  time_point_save(&plan->after_backward[2]);

  // Interpolation in direction 1
  for(size_t z = 0; z < from_info->dims[2]; ++z)
  {
    fftw_complex *positive_start = from + z * from_info->strides[2];
    fftw_execute_dft(plan->n1_backward_real, positive_start, positive_start);
  }

  time_point_save(&plan->after_backward[1]);

  // Interpolation in direction 0
  fftw_execute_dft_c2r(plan->n0_backward_real, from, to);
  time_point_save(&plan->after_backward[0]);
}

static void pa_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out)
{
  pa_plan plan = (pa_plan) detail;
  assert(INTERLEAVED == plan->props.type);

  block_info_t coarse_info, fine_info;
  get_block_info_coarse(&plan->props, &coarse_info);
  get_block_info_fine(&plan->props, &fine_info);

  const size_t block_size = num_elements_block(&coarse_info);
  fftw_complex *const input_copy = rs_alloc_complex(block_size);

  memcpy(input_copy, in, sizeof(fftw_complex) * block_size);
  time_point_save(&plan->before_forward);
  fftw_execute_dft(plan->interleaved_forward, input_copy, input_copy);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(&plan->props, &coarse_info, input_copy);
  pad_coarse_to_fine_interleaved(&plan->props, &coarse_info, input_copy, &fine_info, out, 0);
  time_point_save(&plan->after_padding);
  backward_transform_c2c(plan, &fine_info, out);
  rs_free(input_copy);
}

static void pa_interpolate_real(pa_plan plan, double *in, double *out)
{
  block_info_t transformed_coarse_info, transformed_fine_info, fine_info;
  get_block_info_real_recip_coarse(&plan->props, &transformed_coarse_info);
  get_block_info_real_recip_fine(&plan->props, &transformed_fine_info);
  get_block_info_fine(&plan->props, &fine_info);

  const size_t transformed_size_coarse = num_elements_block(&transformed_coarse_info);
  const size_t transformed_size_fine = num_elements_block(&transformed_fine_info);

  fftw_complex *const scratch_coarse = rs_alloc_complex(transformed_size_coarse);
  fftw_complex *const scratch_fine = rs_alloc_complex(transformed_size_fine);

  time_point_save(&plan->before_forward);
  fftw_execute_dft_r2c(plan->real_forward, in, scratch_coarse);
  time_point_save(&plan->after_forward);
  halve_nyquist_components(&plan->props, &transformed_coarse_info, scratch_coarse);
  pad_coarse_to_fine_interleaved(&plan->props, &transformed_coarse_info, scratch_coarse, &transformed_fine_info, scratch_fine, 1);
  time_point_save(&plan->after_padding);
  backward_transform_c2r(plan, &transformed_fine_info, scratch_fine, &fine_info, out);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

static void pa_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout)
{
  pa_plan plan = (pa_plan) detail;
  assert(SPLIT == plan->props.type || SPLIT_PRODUCT == plan->props.type);

  if (plan->strategy == PACKED)
  {
    block_info_t coarse_info, fine_info;
    get_block_info_coarse(&plan->props, &coarse_info);
    get_block_info_fine(&plan->props, &fine_info);
    const size_t block_size = num_elements_block(&coarse_info);

    fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
    fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

    split_to_interleaved(block_size, rin, iin, scratch_coarse);
    time_point_save(&plan->before_forward);
    fftw_execute_dft(plan->interleaved_forward, scratch_coarse, scratch_coarse);
    time_point_save(&plan->after_forward);
    halve_nyquist_components(&plan->props, &coarse_info, scratch_coarse);
    pad_coarse_to_fine_interleaved(&plan->props, &coarse_info, scratch_coarse, &fine_info, scratch_fine, 0);
    time_point_save(&plan->after_padding);
    backward_transform_c2c(plan, &fine_info, scratch_fine);
    interleaved_to_split(8 * block_size, scratch_fine, rout, iout);

    rs_free(scratch_coarse);
    rs_free(scratch_fine);
  }
  else if (plan->strategy == SEPARATE)
  {
    pa_interpolate_real(plan, rin, rout);
    pa_interpolate_real(plan, iin, iout);
  }
  else
  {
    assert(0 && "Unkown strategy");
  }
}

void pa_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out)
{
  pa_plan plan = (pa_plan) detail;
  assert(SPLIT_PRODUCT == plan->props.type);
  const size_t block_size = num_elements(&plan->props);

  if (plan->strategy == PACKED)
  {
    fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
    fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

    split_to_interleaved(block_size, rin, iin, scratch_coarse);
    pa_interpolate_execute_interleaved(detail, scratch_coarse, scratch_fine);
    complex_to_product(8 * block_size, scratch_fine, out);

    rs_free(scratch_coarse);
    rs_free(scratch_fine);
  }
  else if (plan->strategy == SEPARATE)
  {
    double *const scratch_fine = rs_alloc_real(8 * block_size);
    pa_interpolate_execute_split(detail, rin, iin, out, scratch_fine);
    pointwise_multiply_real(8 * block_size, out, scratch_fine);
    rs_free(scratch_fine);
  }
}

void pa_interpolate_print_timings(const void *detail)
{
  pa_plan plan = (pa_plan) detail;
  printf("Forward: %f\n", time_point_delta(&plan->before_forward, &plan->after_forward));
  printf("Padding: %f\n", time_point_delta(&plan->after_forward, &plan->after_padding));
  for(int dim = 0; dim < 3; ++dim)
    printf("Backward %d: %f\n", dim, time_point_delta(&plan->after_padding, &plan->after_backward[dim]));
}
