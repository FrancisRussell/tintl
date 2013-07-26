#include "common.h"
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <string.h>
#include <interpolate.h>
#include <allocation.h>
#include <fftw_cycle.h>

static void block_copy_coarse_to_fine_interleaved(interpolate_properties_t *props, size_t n0, size_t n1, size_t n2,
    const block_info_t *from_info, const fftw_complex *from,
    const block_info_t *to_info, fftw_complex *to);

void block_copy_coarse_to_fine_interleaved(interpolate_properties_t *props, size_t n0, size_t n1, size_t n2,
    const block_info_t *from_info, const fftw_complex *from,
    const block_info_t *to_info, fftw_complex *to)
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

      from += from_info->strides[1];
      to += to_info->strides[1];
    }

    from += from_info->strides[2] - n1 * from_info->strides[1];
    to += to_info->strides[2] - n1 * to_info->strides[1];
  }
}

void populate_properties(interpolate_properties_t *props, interpolation_t type, size_t n0, size_t n1, size_t n2)
{
  props->type = type;
  props->dims[0] = n2;
  props->dims[1] = n1;
  props->dims[2] = n0;

  props->strides[0] = 1;
  props->strides[1] = n2;
  props->strides[2] = n2 * n1;
}

void halve_nyquist_components(interpolate_properties_t *props, block_info_t *block_info, fftw_complex *coarse)
{
  const size_t n2 = props->dims[2];
  const size_t n1 = props->dims[1];
  const size_t n0 = props->dims[0];

  const size_t d2 = block_info->dims[2];
  const size_t d1 = block_info->dims[1];
  const size_t d0 = block_info->dims[0];

  const size_t s2 = block_info->strides[2];
  const size_t s1 = block_info->strides[1];

  if (n2 % 2 == 0)
    for(size_t i1 = 0; i1 < d1; ++i1)
      for(size_t i0 = 0; i0 < d0; ++i0)
        coarse[s2 * (n2 / 2) +  s1 * i1 + i0] *= 0.5;

  if (n1 % 2 == 0)
    for(size_t i2 = 0; i2 < d2; ++i2)
      for(size_t i0 = 0; i0 < d0; ++i0)
        coarse[s2 * i2 +  s1 * (n1 / 2) + i0] *= 0.5;

  if (n0 % 2 == 0)
    for(size_t i2 = 0; i2 < d2; ++i2)
      for(size_t i1 = 0; i1 < d1; ++i1)
        coarse[s2 * i2 +  s1 * i1 + (n0 / 2)] *= 0.5;
}

void pad_coarse_to_fine_interleaved(interpolate_properties_t *props,
  const block_info_t *from_info, const fftw_complex *from,
  const block_info_t *to_info, fftw_complex *to,
  const int positive_only)
{
  const size_t fine_size = num_elements_block(to_info);
  memset(to, 0, fine_size * sizeof(fftw_complex));

  int corner_flags[3];

  for(corner_flags[2] = 0; corner_flags[2] < 2; ++corner_flags[2])
  {
    for(corner_flags[1] = 0; corner_flags[1] < 2; ++corner_flags[1])
    {
      for(corner_flags[0] = 0; corner_flags[0] < (2 - (positive_only != 0)); ++corner_flags[0])
      {
        const fftw_complex *coarse_block = from;
        fftw_complex *fine_block = to;
        int corner_sizes[3];

        for(int dim = 0; dim < 3; ++dim)
        {
          corner_sizes[dim] = corner_size(props->dims[dim], corner_flags[dim]);
          const int coarse_index = (corner_flags[dim] == 0) ? 0 : from_info->dims[dim] - corner_sizes[dim];
          const int fine_index = (corner_flags[dim] == 0) ? 0 : to_info->dims[dim] - corner_sizes[dim];

          coarse_block += from_info->strides[dim] * coarse_index;
          fine_block += to_info->strides[dim] * fine_index;
        }

        block_copy_coarse_to_fine_interleaved(props, corner_sizes[0], corner_sizes[1], corner_sizes[2],
          from_info, coarse_block, to_info, fine_block);
      }
    }
  }
}

double time_interpolate_split(interpolate_plan plan, const interpolate_properties_t *props)
{
  ticks before, after;
  block_info_t coarse_info;
  get_block_info_coarse(props, &coarse_info);
  const size_t block_size = num_elements_block(&coarse_info);
  double *const in1 = rs_alloc_real(block_size);
  double *const in2 = rs_alloc_real(block_size);
  double *const out1 = rs_alloc_real(8 * block_size);
  double *const out2 = rs_alloc_real(8 * block_size);

  memset(in1, 0, block_size * sizeof(double));
  memset(in2, 0, block_size * sizeof(double));
  memset(out1, 0, block_size * sizeof(double));
  memset(out2, 0, block_size * sizeof(double));

  before = getticks();
  interpolate_execute_split(plan, in1, in2, out1, out2);
  after = getticks();

  rs_free(in1);
  rs_free(in2);
  rs_free(out1);
  rs_free(out2);

  return elapsed(after, before);
}

double time_interpolate_split_product(interpolate_plan plan, const interpolate_properties_t *props)
{
  ticks before, after;
  block_info_t coarse_info;
  get_block_info_coarse(props, &coarse_info);
  const size_t block_size = num_elements_block(&coarse_info);
  double *const in1 = rs_alloc_real(block_size);
  double *const in2 = rs_alloc_real(block_size);
  double *const out = rs_alloc_real(8 * block_size);

  memset(in1, 0, block_size * sizeof(double));
  memset(in2, 0, block_size * sizeof(double));
  memset(out, 0, block_size * sizeof(double));

  before = getticks();
  interpolate_execute_split_product(plan, in1, in2, out);
  after = getticks();

  rs_free(in1);
  rs_free(in2);
  rs_free(out);

  return elapsed(after, before);
}
