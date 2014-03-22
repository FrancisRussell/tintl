#include "common.h"
#include "common_cuda.h"
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

int has_acceptable_cuda_support(void)
{
  int device;
  const enum cudaError error = cudaGetDevice(&device);

  if (error == cudaErrorNoDevice)
    return 0;

  CUDA_CHECK(error);

  struct cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));

  // We need compute capability 1.3 for double precision.
  if (properties.major > 1 || (properties.major == 1 && properties.minor >= 3))
    return 1;
  else
    return 0;
}

__global__ void scale_z(block_info_t input_size, block_info_t block_info, cuDoubleComplex *coarse)
{
  const size_t s2 = block_info.strides[2];
  const size_t s1 = block_info.strides[1];
  const size_t n2 = input_size.dims[2];

  const size_t i1 = blockIdx.x;
  const size_t i0 = threadIdx.x;

  cuDoubleComplex scale = make_cuDoubleComplex(0.5, 0);
  cuDoubleComplex *element = &coarse[s2 * (n2 / 2) +  s1 * i1 + i0];
  *element = cuCmul(*element, scale);
}

__global__ void scale_y(block_info_t input_size, block_info_t block_info, cuDoubleComplex *coarse)
{
  const size_t s2 = block_info.strides[2];
  const size_t s1 = block_info.strides[1];
  const size_t n1 = input_size.dims[1];

  const size_t i2 = blockIdx.x;
  const size_t i0 = threadIdx.x;

  cuDoubleComplex scale = make_cuDoubleComplex(0.5, 0);
  cuDoubleComplex *element = &coarse[s2 * i2 +  s1 * (n1 / 2) + i0];
  *element = cuCmul(*element, scale);
}
__global__ void scale_x(block_info_t input_size, block_info_t block_info, cuDoubleComplex *coarse)
{
  const size_t s2 = block_info.strides[2];
  const size_t s1 = block_info.strides[1];
  const size_t n0 = input_size.dims[2];

  const size_t i2 = blockIdx.x;
  const size_t i1 = threadIdx.x;

  cuDoubleComplex scale = make_cuDoubleComplex(0.5, 0);
  cuDoubleComplex *element = &coarse[s2 * i2 +  s1 * i1 + (n0 / 2)];
  *element = cuCmul(*element, scale);
}

void halve_nyquist_components_cuda(interpolate_plan plan, block_info_t *block_info, cuDoubleComplex *coarse)
{
  const size_t n2 = plan_input_size(plan, 2);
  const size_t n1 = plan_input_size(plan, 1);
  const size_t n0 = plan_input_size(plan, 0);

  const size_t d2 = block_info->dims[2];
  const size_t d1 = block_info->dims[1];
  const size_t d0 = block_info->dims[0];

  if (n2 % 2 == 0)
    scale_z<<<d1, d0>>>(plan->input_size, *block_info, (cuDoubleComplex*) coarse);

  if (n1 % 2 == 0)
    scale_y<<<d2, d1>>>(plan->input_size, *block_info, (cuDoubleComplex*) coarse);

  if (n0 % 2 == 0)
    scale_x<<<d2, d1>>>(plan->input_size, *block_info, (cuDoubleComplex*) coarse);
}

__device__ int calculate_offset(const block_info_t *info, const int *indices)
{
  return info->strides[2] * indices[2] + info->strides[1] * indices[1] + info->strides[0] * indices[0];
}

__global__ void copy_transposed_elements(block_info_t from_info, block_info_t to_info, const cuDoubleComplex *from, cuDoubleComplex *to, int count)
{
  int from_index[3], to_index[3];
  from_index[0] = blockIdx.x * blockDim.x + threadIdx.x;
  from_index[1] = blockIdx.y * blockDim.y + threadIdx.y;
  from_index[2] = blockIdx.z * blockDim.z + threadIdx.z;

  for(int dim = 0; dim < 3; ++dim)
    to_index[(dim + count) % 3] = from_index[dim];

  if (from_index[0] < from_info.dims[0] && from_index[1] < from_info.dims[1] && from_index[2] < from_info.dims[2])
  {
    const int from_offset = calculate_offset(&from_info, from_index);
    const int to_offset = calculate_offset(&to_info, to_index);

    to[to_offset] = from[from_offset];
  }
}

void transpose_block_cuda(const block_info_t *from_info, const cuDoubleComplex *from, block_info_t *to_info, cuDoubleComplex *to, int count)
{
  if (count < 0)
    count = (count % 3) + 3;
  else
    count %= 3;

  assert(count >= 0 && count < 3);

  for(int dim = 0; dim < 3; ++dim)
    to_info->dims[(dim + count) % 3] = from_info->dims[dim];

  populate_strides_unpadded(to_info);

  const dim3 total_dim(from_info->dims[0], from_info->dims[1], from_info->dims[2]);
  const dim3 block_dim(4, 4, 4);
  dim3 grid_dim;

  calcGridDim(total_dim, block_dim, grid_dim);
  copy_transposed_elements<<<grid_dim, block_dim>>>(*from_info, *to_info, from, to, count);
}

__global__ void block_copy_coarse_to_fine_interleaved(
    const block_info_t from_info, const cuDoubleComplex *from,
    const block_info_t to_info, cuDoubleComplex *to, const double scale_factor)
{
  const size_t i2 = blockIdx.y;
  const size_t i1 = blockIdx.x;
  const size_t i0 = threadIdx.x;

  from += from_info.strides[2] * i2 + from_info.strides[1] * i1;
  to += to_info.strides[2] * i2 + to_info.strides[1] * i1;
  to[i0] = cuCmul(from[i0], make_cuDoubleComplex(scale_factor, 0.0));
}


void pad_coarse_to_fine_interleaved_cuda(interpolate_plan plan,
  const block_info_t *from_info, const cuDoubleComplex *from,
  const block_info_t *to_info, cuDoubleComplex *to,
  const int positive_only)
{
  const double scale_factor = 1.0 / num_elements(plan);
  const size_t fine_size = num_elements_block(to_info);
  cudaMemset(to, 0, fine_size * sizeof(cuDoubleComplex));

  int corner_flags[3];

  for(corner_flags[2] = 0; corner_flags[2] < 2; ++corner_flags[2])
  {
    for(corner_flags[1] = 0; corner_flags[1] < 2; ++corner_flags[1])
    {
      for(corner_flags[0] = 0; corner_flags[0] < (2 - (positive_only != 0)); ++corner_flags[0])
      {
        const cuDoubleComplex *coarse_block = from;
        cuDoubleComplex *fine_block = to;
        int corner_sizes[3];

        for(int dim = 0; dim < 3; ++dim)
        {
          corner_sizes[dim] = corner_size(plan_input_size(plan, dim), corner_flags[dim]);
          const int coarse_index = (corner_flags[dim] == 0) ? 0 : from_info->dims[dim] - corner_sizes[dim];
          const int fine_index = (corner_flags[dim] == 0) ? 0 : to_info->dims[dim] - corner_sizes[dim];

          coarse_block += from_info->strides[dim] * coarse_index;
          fine_block += to_info->strides[dim] * fine_index;
        }

        dim3 grid(corner_sizes[1], corner_sizes[2]);
        dim3 block(corner_sizes[0]);
        if (corner_sizes[2] > 0 && corner_sizes[1] > 0 && corner_sizes[0] > 0)
          block_copy_coarse_to_fine_interleaved<<<grid, block>>>(*from_info, coarse_block, *to_info, fine_block, scale_factor);
      }
    }
  }
}

void printCudaDiagnostics(const cudaError_t code, const char *file, const int line)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "A CUDA error code was returned in file %s at line %i: %s\n", file, line, cudaGetErrorString(code));
    exit(EXIT_FAILURE);
  }
}

void printCuFFTDiagnostics(const cufftResult code, const char *file, const int line)
{
  if (code != CUFFT_SUCCESS)
  {
    fprintf(stderr, "A CUFFT error code was returned in file %s at line %i: %d\n", file, line, code);
    exit(EXIT_FAILURE);
  }
}

void calcGridDim(const dim3& total_dim, const dim3& block_dim, dim3& grid_dim)
{
  grid_dim.x = (total_dim.x + block_dim.x - 1) / block_dim.x;
  grid_dim.y = (total_dim.y + block_dim.y - 1) / block_dim.y;
  grid_dim.z = (total_dim.z + block_dim.z - 1) / block_dim.z;
}
