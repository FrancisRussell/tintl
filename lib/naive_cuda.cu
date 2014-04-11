#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/copy.h>
#include "tintl/naive_cuda.h"
#include "tintl/interpolate.h"
#include "tintl/allocation.h"
#include "tintl/timer.h"
#include "common.h"
#include "tintl/forward.h"
#include "common_cuda.h"
#include <complex.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <assert.h>

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

  cufftHandle interleaved_forward;
  cufftHandle interleaved_backward;

  int has_real_plans;
  cufftHandle real_forward;
  cufftHandle real_backward;
} naive_plan_s;


typedef naive_plan_s *naive_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */


static const char *get_name(const interpolate_plan plan);
static void naive_interpolate_execute_interleaved(interpolate_plan plan, rs_complex *in, rs_complex *out);
static void naive_interpolate_execute_split(interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);
static void naive_interpolate_execute_split_product(interpolate_plan plan, double *rin, double *iin, double *out);
static void naive_interpolate_print_timings(const interpolate_plan plan);
static void naive_interpolate_destroy_detail(interpolate_plan plan);
static void naive_set_flags(interpolate_plan plan, const int flags);
static void naive_get_statistic_float(const interpolate_plan plan, int statistic, int index, stat_type_t *type, double *result);

static void plan_common(naive_plan plan, interpolation_t type, int n0, int n1, int n2, int flags);

static const char *get_name(const interpolate_plan plan)
{
  return "naive-cuda";
}

static interpolate_plan allocate_plan(void)
{
  setup_threading();

  interpolate_plan holder = (interpolate_plan) malloc(sizeof(naive_plan_s));
  assert(holder != NULL);

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

static void plan_common(naive_plan plan, interpolation_t type, int n0, int n1, int n2, int flags)
{
  populate_properties((interpolate_plan) plan, type, n0, n1, n2);
  interpolate_plan parent = cast_to_parent(plan);

  block_info_t coarse_info, fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);

  int rev_dims[] = { coarse_info.dims[2], coarse_info.dims[1], coarse_info.dims[0] };
  int rev_fine_dims[] = { fine_info.dims[2], fine_info.dims[1], fine_info.dims[0] };

  CUFFT_CHECK(cufftPlanMany(&plan->interleaved_forward, 3, rev_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1));
  CUFFT_CHECK(cufftPlanMany(&plan->interleaved_backward, 3, rev_fine_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1));

  plan->has_real_plans = 0;
}

interpolate_plan interpolate_plan_3d_naive_cuda_interleaved(int n0, int n1, int n2, int flags)
{
  if (!has_acceptable_cuda_support())
    return NULL;

  interpolate_plan wrapper = allocate_plan();
  naive_plan plan = (naive_plan) wrapper;

  plan_common(plan, INTERPOLATE_INTERLEAVED, n0, n1, n2, flags);
  plan->strategy = PACKED;

  return wrapper;
}

interpolate_plan interpolate_plan_3d_naive_cuda_split(int n0, int n1, int n2, int flags)
{
  if (!has_acceptable_cuda_support())
    return NULL;

  interpolate_plan parent = allocate_plan();
  naive_plan plan = (naive_plan) parent;

  plan_common(plan, INTERPOLATE_SPLIT, n0, n1, n2, flags);

  block_info_t coarse_info, fine_info, transformed_coarse_info, transformed_fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);
  get_block_info_real_recip_coarse(parent, &transformed_coarse_info);
  get_block_info_real_recip_fine(parent, &transformed_fine_info);

  int rev_dims[] = { coarse_info.dims[2], coarse_info.dims[1], coarse_info.dims[0] };
  int rev_fine_dims[] = { fine_info.dims[2], fine_info.dims[1], fine_info.dims[0] };

  CUFFT_CHECK(cufftPlanMany(&plan->real_forward, 3, rev_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_D2Z, 1));
  CUFFT_CHECK(cufftPlanMany(&plan->real_backward, 3, rev_fine_dims, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2D, 1));

  plan->strategy = SEPARATE;
  const double separate_time = time_interpolate_split(parent);
  plan->strategy = PACKED;
  const double packed_time = time_interpolate_split(parent);
  plan->strategy = (separate_time < packed_time) ? SEPARATE : PACKED;

  plan->has_real_plans = 1;

  return parent;
}

interpolate_plan interpolate_plan_3d_naive_cuda_product(int n0, int n1, int n2, int flags)
{
  if (!has_acceptable_cuda_support())
    return NULL;

  interpolate_plan parent = interpolate_plan_3d_naive_cuda_split(n0, n1, n2, flags);
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

  cufftDestroy(plan->interleaved_forward);
  cufftDestroy(plan->interleaved_backward);

  if (plan->has_real_plans)
  {
    cufftDestroy(plan->real_forward);
    cufftDestroy(plan->real_backward);
  }
}

static void naive_interpolate_execute_interleaved(interpolate_plan parent, rs_complex *in, rs_complex *out)
{
  naive_plan plan = (naive_plan) parent;
  assert(plan->strategy == PACKED);

  block_info_t coarse_info, fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);
  const size_t block_size = num_elements_block(&coarse_info);

  thrust::device_vector<cuDoubleComplex> dev_in(block_size);
  thrust::device_vector<cuDoubleComplex> dev_out(block_size * 8);

  CUDA_CHECK(cudaHostRegister(in, sizeof(rs_complex) * block_size, 0));
  CUDA_CHECK(cudaMemcpy(thrust::raw_pointer_cast(&dev_in[0]), in, sizeof(rs_complex) * block_size, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaHostUnregister(in));

  CUFFT_CHECK(cufftExecZ2Z(plan->interleaved_forward, thrust::raw_pointer_cast(&dev_in[0]), thrust::raw_pointer_cast(&dev_in[0]), CUFFT_FORWARD));

  halve_nyquist_components_cuda(parent, &coarse_info, thrust::raw_pointer_cast(&dev_in[0]));
  pad_coarse_to_fine_interleaved_cuda(parent,
    &coarse_info, thrust::raw_pointer_cast(&dev_in[0]), &fine_info, thrust::raw_pointer_cast(&dev_out[0]), 0);

  CUFFT_CHECK(cufftExecZ2Z(plan->interleaved_backward,
    thrust::raw_pointer_cast(&dev_out[0]), thrust::raw_pointer_cast(&dev_out[0]), CUFFT_INVERSE));

  CUDA_CHECK(cudaHostRegister(out, sizeof(rs_complex) * block_size * 8, 0));
  CUDA_CHECK(cudaMemcpy(out, thrust::raw_pointer_cast(&dev_out[0]), sizeof(rs_complex) * block_size * 8, cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaHostUnregister(out));

  CUDA_CHECK(cudaDeviceSynchronize());
}

static void naive_interpolate_real(naive_plan plan, double *in, const thrust::device_ptr<double>& dev_out)
{
  block_info_t coarse_info, transformed_coarse_info, transformed_fine_info;
  interpolate_plan parent = cast_to_parent(plan);
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_real_recip_coarse(parent, &transformed_coarse_info);
  get_block_info_real_recip_fine(parent, &transformed_fine_info);

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

  halve_nyquist_components_cuda(parent, &transformed_coarse_info, thrust::raw_pointer_cast(&scratch_coarse[0]));
  pad_coarse_to_fine_interleaved_cuda(parent,
    &transformed_coarse_info, thrust::raw_pointer_cast(&scratch_coarse[0]), &transformed_fine_info, thrust::raw_pointer_cast(&scratch_fine[0]), 1);

  CUFFT_CHECK(cufftExecZ2D(plan->real_backward, thrust::raw_pointer_cast(&scratch_fine[0]), thrust::raw_pointer_cast(dev_out)));
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

    rs_complex *const scratch_coarse = tintl_alloc_complex(block_size);
    rs_complex *const scratch_fine = tintl_alloc_complex(8 * block_size);
    assert(scratch_coarse != NULL);
    assert(scratch_fine != NULL);

    interleave_real(block_size, (double*) scratch_coarse, rin, iin);
    naive_interpolate_execute_interleaved(parent, scratch_coarse, scratch_fine);
    deinterleave_real(8 * block_size, (const double*) scratch_fine, rout, iout);

    tintl_free(scratch_fine);
    tintl_free(scratch_coarse);
  }
  else if (plan->strategy == SEPARATE)
  {
    block_info_t coarse_info;
    get_block_info_coarse(parent, &coarse_info);
    const size_t block_size = num_elements_block(&coarse_info);

    CUDA_CHECK(cudaHostRegister(rout, sizeof(double) * block_size * 8, 0));
    CUDA_CHECK(cudaHostRegister(iout, sizeof(double) * block_size * 8, 0));

    thrust::device_vector<double> dev_out_r(block_size * 8);
    thrust::device_vector<double> dev_out_i(block_size * 8);

    naive_interpolate_real(plan, rin, &dev_out_r[0]);
    naive_interpolate_real(plan, iin, &dev_out_i[0]);

    CUDA_CHECK(cudaMemcpy(rout, thrust::raw_pointer_cast(&dev_out_r[0]), sizeof(double) * block_size * 8, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(iout, thrust::raw_pointer_cast(&dev_out_i[0]), sizeof(double) * block_size * 8, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaHostUnregister(rout));
    CUDA_CHECK(cudaHostUnregister(iout));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else
  {
    assert(0 && "Unknown strategy.");
  }
}

static void naive_interpolate_execute_split_product(interpolate_plan parent, double *rin, double *iin, double *out)
{
  naive_plan plan = (naive_plan) parent;
  assert(INTERPOLATE_SPLIT_PRODUCT == parent->type);
  const size_t block_size = num_elements(parent);

  if (plan->strategy == PACKED)
  {
    rs_complex *const scratch_coarse = tintl_alloc_complex(block_size);
    rs_complex *const scratch_fine = tintl_alloc_complex(8 * block_size);
    assert(scratch_coarse != NULL);
    assert(scratch_fine != NULL);

    interleave_real(block_size, (double*) scratch_coarse, rin, iin);
    naive_interpolate_execute_interleaved(parent, scratch_coarse, scratch_fine);
    complex_to_product(8 * block_size, scratch_fine, out);

    tintl_free(scratch_coarse);
    tintl_free(scratch_fine);
  }
  if (plan->strategy == SEPARATE)
  {
    block_info_t coarse_info;
    get_block_info_coarse(parent, &coarse_info);
    const size_t block_size = num_elements_block(&coarse_info);

    CUDA_CHECK(cudaHostRegister(rin, sizeof(double) * block_size, 0));
    CUDA_CHECK(cudaHostRegister(iin, sizeof(double) * block_size, 0));
    CUDA_CHECK(cudaHostRegister(out, sizeof(double) * block_size * 8, 0));

    thrust::device_vector<double> dev_out_r(block_size * 8);
    thrust::device_vector<double> dev_out_i(block_size * 8);

    naive_interpolate_real(plan, rin, &dev_out_r[0]);
    naive_interpolate_real(plan, iin, &dev_out_i[0]);

    CUDA_CHECK(cudaHostUnregister(rin));
    CUDA_CHECK(cudaHostUnregister(iin));

    thrust::transform(dev_out_r.begin(), dev_out_r.end(), dev_out_i.begin(), dev_out_r.begin(), thrust::plus<double>());

    CUDA_CHECK(cudaMemcpy(out, thrust::raw_pointer_cast(&dev_out_r[0]), sizeof(double) * block_size * 8, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaHostUnregister(out));
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  else
  {
    assert(0 && "Unknown strategy");
  }
}

static void naive_interpolate_print_timings(const interpolate_plan parent)
{
}

