#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include "interpolate.h"
#include "padding_aware_cuda.h"
#include "timer.h"
#include "allocation.h"
#include "common.h"
#include "forward.h"
#include "fftw_cycle.h"
#include "common_cuda.h"
#include <complex.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include <cufft.h>
#include <cuda_runtime_api.h>

typedef enum
{
  PACKED,
  SEPARATE
} strategy_t;

/// Implementation-specific structure for padding-aware interpolation plans.
typedef struct
{
  interpolate_properties_t props;
  strategy_t strategy;

  cufftHandle interleaved_forward;
  cufftHandle n2_backward_interleaved[2];
  cufftHandle n1_backward_interleaved[2];
  cufftHandle n0_backward_interleaved;

  int n2_backward_interleaved_needed[2];
  int n1_backward_interleaved_needed[2];

  cufftHandle real_forward;
  cufftHandle n2_backward_real;
  cufftHandle n1_backward_real;
  cufftHandle n0_backward_real;

  int has_real_plans;

  time_point_t before;
  time_point_t after;
} pa_plan_s;

typedef pa_plan_s *pa_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */
static const char *get_name(const void *detail);
static void pa_interpolate_execute_interleaved(const void *detail, rs_complex *in, rs_complex *out);
static void pa_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout);
static void pa_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out);
static void pa_interpolate_print_timings(const void *detail);
static void pa_interpolate_destroy_detail(void *detail);

static void plan_common(pa_plan plan, interpolation_t type, int n0, int n1, int n2, int flags);

static const char *get_name(const void *detail)
{
  return "padding-aware-cuda";
}

static interpolate_plan allocate_plan(void)
{
  setup_threading();

  interpolate_plan holder = (interpolate_plan) malloc(sizeof(interpolate_plan_s));
  assert(holder != NULL);

  holder->ref_cnt = 1;

  holder->detail = malloc(sizeof(pa_plan_s));
  assert(holder->detail != NULL);

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

  block_info_t fine_info;
  get_block_info_fine(&plan->props, &fine_info);

  int rev_dims[] = { plan->props.dims[2], plan->props.dims[1], plan->props.dims[0] };

  CUFFT_CHECK(cufftPlanMany(&plan->interleaved_forward, 3, rev_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1));

  // For small FFT sizes, some corners may be size zero, which CUFFT dislikes.
  for(int corner = 0; corner < 2; ++corner)
  {
    plan->n2_backward_interleaved_needed[corner] = (corner_size(plan->props.dims[0], corner) != 0);
    plan->n1_backward_interleaved_needed[corner] = (corner_size(plan->props.dims[0], corner) != 0);
  }

  // Interpolation in direction 2, iteration in direction 0, positive frequencies
  if (plan->n2_backward_interleaved_needed[0])
  {
    CUFFT_CHECK(cufftPlanMany(&plan->n2_backward_interleaved[0], 1, &fine_info.dims[2],
        &fine_info.strides[2], fine_info.strides[2], fine_info.strides[0],
        &fine_info.strides[2], fine_info.strides[2], fine_info.strides[0],
        CUFFT_Z2Z, corner_size(plan->props.dims[0], 0)));
  }

  // Interpolation in direction 2, iteration in direction 0, negative frequencies
  if (plan->n2_backward_interleaved_needed[1])
  {
    CUFFT_CHECK(cufftPlanMany(&plan->n2_backward_interleaved[1], 1, &fine_info.dims[2],
        &fine_info.dims[2], fine_info.strides[2], fine_info.strides[0],
        &fine_info.dims[2], fine_info.strides[2], fine_info.strides[0],
        CUFFT_Z2Z, corner_size(plan->props.dims[0], 1)));
  }

  // Interpolation in direction 1, iteration in direction 0, positive frequencies
  if (plan->n1_backward_interleaved_needed[0])
  {
    CUFFT_CHECK(cufftPlanMany(&plan->n1_backward_interleaved[0], 1, &fine_info.dims[1],
        &fine_info.dims[1], fine_info.strides[1], fine_info.strides[0],
        &fine_info.dims[1], fine_info.strides[1], fine_info.strides[0],
        CUFFT_Z2Z, corner_size(plan->props.dims[0], 0)));
  }

  // Interpolation in direction 1, iteration in direction 0, negative frequencies
  if (plan->n1_backward_interleaved_needed[1])
  {
    CUFFT_CHECK(cufftPlanMany(&plan->n1_backward_interleaved[1], 1, &fine_info.dims[1],
        &fine_info.dims[1], fine_info.strides[1], fine_info.strides[0],
        &fine_info.dims[1], fine_info.strides[1], fine_info.strides[0],
        CUFFT_Z2Z, corner_size(plan->props.dims[0], 1)));
  }

  // Interpolation in direction 0, iteration in direction 1, all frequencies
  CUFFT_CHECK(cufftPlanMany(&plan->n0_backward_interleaved, 1, &fine_info.dims[0],
      &fine_info.dims[0], fine_info.strides[0], fine_info.strides[1],
      &fine_info.dims[0], fine_info.strides[0], fine_info.strides[1],
      CUFFT_Z2Z, fine_info.dims[1] * fine_info.dims[2]));

  plan->has_real_plans = 0;
}

interpolate_plan interpolate_plan_3d_padding_aware_cuda_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  pa_plan plan = (pa_plan) wrapper->detail;

  plan_common(plan, INTERLEAVED, n0, n1, n2, flags);
  plan->strategy = PACKED;

  return wrapper;
}

interpolate_plan interpolate_plan_3d_padding_aware_cuda_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  pa_plan plan = (pa_plan) wrapper->detail;

  plan_common(plan, SPLIT, n0, n1, n2, flags);

  block_info_t coarse_info, fine_info, transformed_coarse_info, transformed_fine_info;
  get_block_info_coarse(&plan->props, &coarse_info);
  get_block_info_fine(&plan->props, &fine_info);
  get_block_info_real_recip_coarse(&plan->props, &transformed_coarse_info);
  get_block_info_real_recip_fine(&plan->props, &transformed_fine_info);

  int rev_dims[] = { plan->props.dims[2], plan->props.dims[1], plan->props.dims[0] };

  CUFFT_CHECK(cufftPlanMany(&plan->real_forward, 3, rev_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, 1));

  // Interpolation in direction 2, iteration in direction 0, positive frequencies
  if (plan->n2_backward_interleaved_needed[0])
  {
    CUFFT_CHECK(cufftPlanMany(&plan->n2_backward_real, 1, &transformed_fine_info.dims[2],
      &transformed_fine_info.dims[2], transformed_fine_info.strides[2], transformed_fine_info.strides[0],
      &transformed_fine_info.dims[2], transformed_fine_info.strides[2], transformed_fine_info.strides[0],
      CUFFT_Z2Z, corner_size(plan->props.dims[0], 0)));
  }

  // Interpolation in direction 1, iteration in direction 0, positive frequencies
  if (plan->n2_backward_interleaved_needed[0])
  {
    CUFFT_CHECK(cufftPlanMany(&plan->n1_backward_real, 1, &transformed_fine_info.dims[1],
        &transformed_fine_info.dims[1], transformed_fine_info.strides[1], transformed_fine_info.strides[0],
        &transformed_fine_info.dims[1], transformed_fine_info.strides[1], transformed_fine_info.strides[0],
        CUFFT_Z2Z, corner_size(plan->props.dims[0], 0)));
  }

  // Interpolation in direction 0, iteration in direction 1, all frequencies
  CUFFT_CHECK(cufftPlanMany(&plan->n0_backward_real, 1, &fine_info.dims[0], 
      &fine_info.dims[0], transformed_fine_info.strides[0], transformed_fine_info.strides[1],
      &fine_info.dims[0], fine_info.strides[0],             fine_info.strides[1],
      CUFFT_Z2D, fine_info.dims[1] * fine_info.dims[2]));

  plan->has_real_plans = 1;

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split(wrapper, plan->props.dims);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split(wrapper, plan->props.dims);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;

  return wrapper;
}

interpolate_plan interpolate_plan_3d_padding_aware_cuda_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = interpolate_plan_3d_padding_aware_cuda_split(n0, n1, n2, flags);
  pa_plan plan = (pa_plan) wrapper->detail;
  plan->props.type = SPLIT_PRODUCT;

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split_product(wrapper, plan->props.dims);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split_product(wrapper, plan->props.dims);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;

  return wrapper;
}

static void pa_interpolate_destroy_detail(void *detail)
{
  pa_plan plan = (pa_plan) detail;

  cufftDestroy(plan->interleaved_forward);

  for(int corner = 0; corner < 2; ++corner)
  {
    if (plan->n2_backward_interleaved_needed[corner])
      cufftDestroy(plan->n2_backward_interleaved[corner]);

    if (plan->n1_backward_interleaved_needed[corner])
      cufftDestroy(plan->n1_backward_interleaved[corner]);
  }

  cufftDestroy(plan->n0_backward_interleaved);

  if (plan->has_real_plans)
  {
    cufftDestroy(plan->real_forward);
    cufftDestroy(plan->n2_backward_real);
    cufftDestroy(plan->n1_backward_real);
    cufftDestroy(plan->n0_backward_real);
  }

  free(plan);
}

static void backward_transform_c2c(const pa_plan plan, const block_info_t *data_info, cuDoubleComplex *data)
{
  size_t corner_sizes[3][2];

  for(int negative = 0; negative < 2; ++negative)
    for(int dim = 0; dim < 3; ++dim)
      corner_sizes[dim][negative] = corner_size(plan->props.dims[dim], negative);

  // Interpolation in direction 2
  for(size_t y = 0; y < corner_sizes[1][0]; ++y)
  {
    cuDoubleComplex *positive_start = data + y * data_info->strides[1];
    if (plan->n2_backward_interleaved_needed[0])
      CUFFT_CHECK(cufftExecZ2Z(plan->n2_backward_interleaved[0], positive_start, positive_start, CUFFT_INVERSE));

    cuDoubleComplex *negative_start = data + (y + 1) * data_info->strides[1] - corner_sizes[0][1];
    if (plan->n2_backward_interleaved_needed[1])
      CUFFT_CHECK(cufftExecZ2Z(plan->n2_backward_interleaved[1], negative_start, negative_start, CUFFT_INVERSE));
  }

  for(size_t y = data_info->dims[1] - corner_sizes[1][1]; y < data_info->dims[1]; ++y)
  {
    cuDoubleComplex *positive_start = data + y * data_info->strides[1];
    if (plan->n2_backward_interleaved_needed[0])
      CUFFT_CHECK(cufftExecZ2Z(plan->n2_backward_interleaved[0], positive_start, positive_start, CUFFT_INVERSE));

    cuDoubleComplex *negative_start = data + (y + 1) * data_info->strides[1] - corner_sizes[0][1];
    if (plan->n2_backward_interleaved_needed[1])
      CUFFT_CHECK(cufftExecZ2Z(plan->n2_backward_interleaved[1], negative_start, negative_start, CUFFT_INVERSE));
  }

  // Interpolation in direction 1
  for(size_t z = 0; z < data_info->dims[2]; ++z)
  {
    cuDoubleComplex *positive_start = data + z * data_info->strides[2];
    if (plan->n1_backward_interleaved_needed[0])
      CUFFT_CHECK(cufftExecZ2Z(plan->n1_backward_interleaved[0], positive_start, positive_start, CUFFT_INVERSE));

    cuDoubleComplex *negative_start = data + z * data_info->strides[2] + data_info->strides[1] - corner_sizes[0][1];
    if (plan->n1_backward_interleaved_needed[1])
      CUFFT_CHECK(cufftExecZ2Z(plan->n1_backward_interleaved[1], negative_start, negative_start, CUFFT_INVERSE));
  }

  // Interpolation in direction 0
  CUFFT_CHECK(cufftExecZ2Z(plan->n0_backward_interleaved, data, data, CUFFT_INVERSE));
}

static void backward_transform_c2r(const pa_plan plan,
  const block_info_t *from_info, cuDoubleComplex *from,
  const block_info_t *to_info, double *to)
{
  size_t corner_sizes[3][2];

  for(int negative = 0; negative < 2; ++negative)
    for(int dim = 0; dim < 3; ++dim)
      corner_sizes[dim][negative] = corner_size(plan->props.dims[dim], negative);

  // Interpolation in direction 2
  for(size_t y = 0; y < corner_sizes[1][0]; ++y)
  {
    cuDoubleComplex *positive_start = from + y * from_info->strides[1];
    if (plan->n2_backward_interleaved_needed[0])
      CUFFT_CHECK(cufftExecZ2Z(plan->n2_backward_real, positive_start, positive_start, CUFFT_INVERSE));
  }

  for(size_t y = to_info->dims[1] - corner_sizes[1][1]; y < to_info->dims[1]; ++y)
  {
    cuDoubleComplex *positive_start = from + y * from_info->strides[1];
    if (plan->n2_backward_interleaved_needed[0])
      CUFFT_CHECK(cufftExecZ2Z(plan->n2_backward_real, positive_start, positive_start, CUFFT_INVERSE));
  }

  // Interpolation in direction 1
  for(size_t z = 0; z < from_info->dims[2]; ++z)
  {
    cuDoubleComplex *positive_start = from + z * from_info->strides[2];
    if (plan->n1_backward_interleaved_needed[0])
      CUFFT_CHECK(cufftExecZ2Z(plan->n1_backward_real, positive_start, positive_start, CUFFT_INVERSE));
  }

  // Interpolation in direction 0
  CUFFT_CHECK(cufftExecZ2D(plan->n0_backward_real, from, to));
}

static void pa_interpolate_execute_interleaved(const void *detail, rs_complex *in, rs_complex *out)
{
  pa_plan plan = (pa_plan) detail;
  assert(plan->strategy == PACKED);

  block_info_t coarse_info, fine_info;
  get_block_info_coarse(&plan->props, &coarse_info);
  get_block_info_fine(&plan->props, &fine_info);
  const size_t block_size = num_elements_block(&coarse_info);

  thrust::device_vector<cuDoubleComplex> dev_in(block_size);
  thrust::device_vector<cuDoubleComplex> dev_out(block_size * 8);

  CUDA_CHECK(cudaHostRegister(in, sizeof(rs_complex) * block_size, 0));
  CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(&dev_in[0]), in, sizeof(rs_complex) * block_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaHostUnregister(in));

  CUFFT_CHECK(cufftExecZ2Z(plan->interleaved_forward, thrust::raw_pointer_cast(&dev_in[0]), thrust::raw_pointer_cast(&dev_in[0]), CUFFT_FORWARD));

  halve_nyquist_components_cuda(&plan->props, &coarse_info, thrust::raw_pointer_cast(&dev_in[0]));
  pad_coarse_to_fine_interleaved_cuda(&plan->props, 
    &coarse_info, thrust::raw_pointer_cast(&dev_in[0]), &fine_info, thrust::raw_pointer_cast(&dev_out[0]), 0);
  
  backward_transform_c2c(plan, &fine_info, thrust::raw_pointer_cast(&dev_out[0]));

  CUDA_CHECK(cudaHostRegister(out, sizeof(rs_complex) * block_size * 8, 0));
  CUDA_CHECK(cudaMemcpy(out, thrust::raw_pointer_cast(&dev_out[0]), sizeof(rs_complex) * block_size * 8, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaHostUnregister(out));
  CUDA_CHECK(cudaDeviceSynchronize());
}

static void pa_interpolate_real(pa_plan plan, double *in, const thrust::device_ptr<double>& dev_out)
{
  block_info_t coarse_info, fine_info, transformed_coarse_info, transformed_fine_info;
  get_block_info_coarse(&plan->props, &coarse_info);
  get_block_info_fine(&plan->props, &fine_info);
  get_block_info_real_recip_coarse(&plan->props, &transformed_coarse_info);
  get_block_info_real_recip_fine(&plan->props, &transformed_fine_info);

  const size_t block_size = num_elements_block(&coarse_info);
  const size_t transformed_size_coarse = num_elements_block(&transformed_coarse_info);
  const size_t transformed_size_fine = num_elements_block(&transformed_fine_info);

  thrust::device_vector<double> dev_in(block_size);
  thrust::device_vector<cuDoubleComplex> scratch_coarse(transformed_size_coarse);
  thrust::device_vector<cuDoubleComplex> scratch_fine(transformed_size_fine);

  CUDA_CHECK(cudaHostRegister(in, sizeof(double) * block_size, 0));
  CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(&dev_in[0]), in, sizeof(double) * block_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaHostUnregister(in));

  CUFFT_CHECK(cufftExecD2Z(plan->real_forward, thrust::raw_pointer_cast(&dev_in[0]), thrust::raw_pointer_cast(&scratch_coarse[0])));

  halve_nyquist_components_cuda(&plan->props, &transformed_coarse_info, thrust::raw_pointer_cast(&scratch_coarse[0]));
  pad_coarse_to_fine_interleaved_cuda(&plan->props, 
    &transformed_coarse_info, thrust::raw_pointer_cast(&scratch_coarse[0]), 
    &transformed_fine_info,   thrust::raw_pointer_cast(&scratch_fine[0]), 1);

  backward_transform_c2r(plan, &transformed_fine_info, thrust::raw_pointer_cast(&scratch_fine[0]), 
    &fine_info, thrust::raw_pointer_cast(dev_out));
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

    rs_complex *const scratch_coarse = rs_alloc_complex(block_size);
    rs_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

    interleave_real(block_size, (double*) scratch_coarse, rin, iin);
    pa_interpolate_execute_interleaved(detail, scratch_coarse, scratch_fine);
    deinterleave_real(8 * block_size, (const double*) scratch_fine, rout, iout);

    rs_free(scratch_fine);
    rs_free(scratch_coarse);
  }
  else if (plan->strategy == SEPARATE)
  {
    block_info_t fine_info;
    get_block_info_fine(&plan->props, &fine_info);
    const size_t fine_block_size = num_elements_block(&fine_info);

    thrust::device_vector<double> dev_out_r(fine_block_size);
    thrust::device_vector<double> dev_out_i(fine_block_size);

    pa_interpolate_real(plan, rin, &dev_out_r[0]);
    pa_interpolate_real(plan, iin, &dev_out_i[0]);

    CUDA_CHECK(cudaHostRegister(rout, sizeof(double) * fine_block_size, 0));
    CUDA_CHECK(cudaMemcpy(rout, thrust::raw_pointer_cast(&dev_out_r[0]), sizeof(double) * fine_block_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaHostUnregister(rout));

    CUDA_CHECK(cudaHostRegister(iout, sizeof(double) * fine_block_size, 0));
    CUDA_CHECK(cudaMemcpy(iout, thrust::raw_pointer_cast(&dev_out_i[0]), sizeof(double) * fine_block_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaHostUnregister(iout));

    CUDA_CHECK(cudaDeviceSynchronize());
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
    rs_complex *const scratch_coarse = rs_alloc_complex(block_size);
    rs_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

    interleave_real(block_size, (double*) scratch_coarse, rin, iin);
    pa_interpolate_execute_interleaved(detail, scratch_coarse, scratch_fine);
    complex_to_product(8 * block_size, scratch_fine, out);

    rs_free(scratch_coarse);
    rs_free(scratch_fine);
  }
  else if (plan->strategy == SEPARATE)
  {
    block_info_t fine_info;
    get_block_info_fine(&plan->props, &fine_info);
    const size_t fine_block_size = num_elements_block(&fine_info);

    thrust::device_vector<double> dev_out_r(fine_block_size);
    thrust::device_vector<double> dev_out_i(fine_block_size);

    pa_interpolate_real(plan, rin, &dev_out_r[0]);
    pa_interpolate_real(plan, iin, &dev_out_i[0]);

    thrust::transform(dev_out_r.begin(), dev_out_r.end(), dev_out_i.begin(), dev_out_r.begin(), thrust::plus<double>());

    CUDA_CHECK(cudaHostRegister(out, sizeof(double) * block_size * 8, 0));
    CUDA_CHECK(cudaMemcpy(out, thrust::raw_pointer_cast(&dev_out_r[0]), sizeof(double) * block_size * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaHostUnregister(out));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}

void pa_interpolate_print_timings(const void *detail)
{
  /*
  pa_plan plan = (pa_plan) detail;
  printf("Forward: %f\n", time_point_delta(&plan->before_forward, &plan->after_forward));
  printf("Padding: %f\n", time_point_delta(&plan->after_forward, &plan->after_padding));
  for(int dim = 0; dim < 3; ++dim)
    printf("Backward %d: %f\n", dim, time_point_delta(&plan->after_padding, &plan->after_backward[dim]));
  */
}
