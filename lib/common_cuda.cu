#include "common.h"
#include <cuComplex.h>
#include "common_cuda.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void scale_z(interpolate_properties_t props, block_info_t block_info, cuDoubleComplex *coarse)
{
  const size_t s2 = block_info.strides[2];
  const size_t s1 = block_info.strides[1];
  const size_t n2 = props.dims[2];

  const size_t i1 = blockIdx.x;
  const size_t i0 = threadIdx.x;

  cuDoubleComplex scale = make_cuDoubleComplex(0.5, 0);
  cuDoubleComplex *element = &coarse[s2 * (n2 / 2) +  s1 * i1 + i0];
  *element = cuCmul(*element, scale);
}

__global__ void scale_y(interpolate_properties_t props, block_info_t block_info, cuDoubleComplex *coarse)
{
  const size_t s2 = block_info.strides[2];
  const size_t s1 = block_info.strides[1];
  const size_t n1 = props.dims[1];

  const size_t i2 = blockIdx.x;
  const size_t i0 = threadIdx.x;

  cuDoubleComplex scale = make_cuDoubleComplex(0.5, 0);
  cuDoubleComplex *element = &coarse[s2 * i2 +  s1 * (n1 / 2) + i0];
  *element = cuCmul(*element, scale);
}
__global__ void scale_x(interpolate_properties_t props, block_info_t block_info, cuDoubleComplex *coarse)
{
  const size_t s2 = block_info.strides[2];
  const size_t s1 = block_info.strides[1];
  const size_t n0 = props.dims[2];

  const size_t i2 = blockIdx.x;
  const size_t i1 = threadIdx.x;

  cuDoubleComplex scale = make_cuDoubleComplex(0.5, 0);
  cuDoubleComplex *element = &coarse[s2 * i2 +  s1 * i1 + (n0 / 2)];
  *element = cuCmul(*element, scale);
}

void halve_nyquist_components_cuda(interpolate_properties_t *props, block_info_t *block_info, cuDoubleComplex *coarse)
{
  const size_t n2 = props->dims[2];
  const size_t n1 = props->dims[1];
  const size_t n0 = props->dims[0];

  const size_t d2 = block_info->dims[2];
  const size_t d1 = block_info->dims[1];
  const size_t d0 = block_info->dims[0];

  if (n2 % 2 == 0)
    scale_z<<<d1, d0>>>(*props, *block_info, (cuDoubleComplex*) coarse);

  dim3 ySlice(d0, d2);
  if (n1 % 2 == 0)
    scale_y<<<d2, d1>>>(*props, *block_info, (cuDoubleComplex*) coarse);

  if (n0 % 2 == 0)
    scale_x<<<d2, d1>>>(*props, *block_info, (cuDoubleComplex*) coarse);
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


void pad_coarse_to_fine_interleaved_cuda(interpolate_properties_t *props,
  const block_info_t *from_info, const cuDoubleComplex *from,
  const block_info_t *to_info, cuDoubleComplex *to,
  const int positive_only)
{
  const double scale_factor = 1.0 / num_elements(props);
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
          corner_sizes[dim] = corner_size(props->dims[dim], corner_flags[dim]);
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
