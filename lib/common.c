#include "common.h"
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <string.h>

static void block_copy_coarse_to_fine_interleaved(interpolate_properties_t *props, size_t n0, size_t n1, size_t n2, const fftw_complex *from, fftw_complex *to);

void block_copy_coarse_to_fine_interleaved(interpolate_properties_t *props, size_t n0, size_t n1, size_t n2, const fftw_complex *from, fftw_complex *to)
{
  assert(n0 <= props->dims[0]);
  assert(n1 <= props->dims[1]);
  assert(n2 <= props->dims[2]);

  const double scale_factor = 1.0 / num_elements(props);

  for(size_t i2=0; i2 < n2; ++i2)
  {
    for(size_t i1=0; i1 < n1; ++i1)
    {
      for(size_t i0=0; i0 < n0; ++i0)
        to[i0] = from[i0] * scale_factor;

      from += props->strides[1];
      to += props->fine_strides[1];
    }

    from += props->strides[2] - n1 * props->strides[1];
    to += props->fine_strides[2] - n1 * props->fine_strides[1];
  }
}

void populate_properties(interpolate_properties_t *props, interpolation_t type, size_t n0, size_t n1, size_t n2)
{
  props->type = type;
  props->dims[0] = n2;
  props->dims[1] = n1;
  props->dims[2] = n0;

  for(size_t dim = 0; dim < 3; ++dim)
    props->fine_dims[dim] = props->dims[dim] * 2;

  props->strides[0] = 1;
  props->strides[1] = n2;
  props->strides[2] = n2 * n1;

  props->fine_strides[0] = 1;
  props->fine_strides[1] = n2 * 2;
  props->fine_strides[2] = n2 * n1 * 4;
}


void halve_nyquist_components(interpolate_properties_t *props, fftw_complex *coarse)
{
  const size_t n2 = props->dims[2];
  const size_t n1 = props->dims[1];
  const size_t n0 = props->dims[0];

  const size_t s2 = props->strides[2];
  const size_t s1 = props->strides[1];

  if (n2 % 2 == 0)
    for(size_t i1 = 0; i1 < n1; ++i1)
      for(size_t i0 = 0; i0 < n0; ++i0)
        coarse[s2 * (n2 / 2) +  s1 * i1 + i0] *= 0.5;

  if (n1 % 2 == 0)
    for(size_t i2 = 0; i2 < n2; ++i2)
      for(size_t i0 = 0; i0 < n0; ++i0)
        coarse[s2 * i2 +  s1 * (n1 / 2) + i0] *= 0.5;

  if (n0 % 2 == 0)
    for(size_t i2 = 0; i2 < n2; ++i2)
      for(size_t i1 = 0; i1 < n1; ++i1)
        coarse[s2 * i2 +  s1 * i1 + (n0 / 2)] *= 0.5;
}

void pad_coarse_to_fine_interleaved(interpolate_properties_t *props, const fftw_complex *from, fftw_complex *to)
{
  const size_t coarse_size = num_elements(props);
  memset(to, 0, 8 * coarse_size * sizeof(fftw_complex));

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
          corner_sizes[dim] = corner_size(props->dims[dim], corner_flags[dim]);
          const int coarse_index = (corner_flags[dim] == 0) ? 0 : props->dims[dim] - corner_sizes[dim];
          const int fine_index = (corner_flags[dim] == 0) ? 0 : props->fine_dims[dim] - corner_sizes[dim];

          coarse_block += props->strides[dim] * coarse_index;
          fine_block += props->fine_strides[dim] * fine_index;
        }

        block_copy_coarse_to_fine_interleaved(props, corner_sizes[0], corner_sizes[1], corner_sizes[2], coarse_block, fine_block);
      }
    }
  }
}
