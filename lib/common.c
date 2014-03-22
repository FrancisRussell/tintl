#include "common.h"
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <string.h>
#include <interpolate.h>
#include <allocation.h>
#include <fftw_cycle.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef FFTW_IS_MKL
#include <fftw3_mkl.h>
#endif


/// Copy complex values between 3D arrays of different strides and
/// simultaneously scale. Values are scaled by the inverse of the number
/// of source elements being interpolated.
/// @param props properties of the interpolation.
/// @param n0 block width in dimension 0.
/// @param n1 block width in dimension 1.
/// @param n2 block width in dimension 2.
/// @param from_info size/stride details of source array.
/// @param to_info size/stride details of target array.
/// @param from source array.
/// @param to target array.
static void block_copy_coarse_to_fine_interleaved(interpolate_plan plan, size_t n0, size_t n1, size_t n2,
    const block_info_t *from_info, const fftw_complex *from,
    const block_info_t *to_info, fftw_complex *to);

void block_copy_coarse_to_fine_interleaved(interpolate_plan plan, size_t n0, size_t n1, size_t n2,
    const block_info_t *from_info, const fftw_complex *from,
    const block_info_t *to_info, fftw_complex *to)
{
  assert(n0 <= plan->input_size.dims[0]);
  assert(n1 <= plan->input_size.dims[1]);
  assert(n2 <= plan->input_size.dims[2]);

  const double scale_factor = 1.0 / num_elements(plan);

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


/// Populate an interpolate properties structure given the size of the interpolation.
void populate_properties(interpolate_plan plan, interpolation_t type, size_t n0, size_t n1, size_t n2)
{
  plan->magic = interpolate_plan_magic_value;
  plan->type = type;
  plan->input_size.dims[0] = n2;
  plan->input_size.dims[1] = n1;
  plan->input_size.dims[2] = n0;

  plan->input_size.strides[0] = 1;
  plan->input_size.strides[1] = n2;
  plan->input_size.strides[2] = n2 * n1;
}

/// Halve the magnitude of Nyquist components in a frequency domain
/// representation of a data-set. This is necessary when performing
/// interpolation when one or more dimensions is even. It is valid for
/// the block to be truncated, as occurs with r2c transforms.
/// @param props interpolation properties.
/// @param block_info block information of the frequency representation.
/// @param coarse the data
void halve_nyquist_components(interpolate_plan plan, block_info_t *block_info, fftw_complex *coarse)
{
  const size_t n2 = plan_input_size(plan, 2);
  const size_t n1 = plan_input_size(plan, 1);
  const size_t n0 = plan_input_size(plan, 0);

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


/// Pads the frequency domain representation of a data-set to double
/// its density in three dimensions. In the case of even dimensions, the
/// Nyquist component is duplicated.
/// @param props interpolation properties.
/// @param from_info source array information.
/// @param to_info target array information.
/// @param from source array.
/// @param to target array.
/// @param positive_only if non-zero, do not copy negative frequency
/// blocks in the contiguous dimension.
void pad_coarse_to_fine_interleaved(interpolate_plan plan,
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
          corner_sizes[dim] = corner_size(plan_input_size(plan, dim), corner_flags[dim]);
          const int coarse_index = (corner_flags[dim] == 0) ? 0 : from_info->dims[dim] - corner_sizes[dim];
          const int fine_index = (corner_flags[dim] == 0) ? 0 : to_info->dims[dim] - corner_sizes[dim];

          coarse_block += from_info->strides[dim] * coarse_index;
          fine_block += to_info->strides[dim] * fine_index;
        }

        block_copy_coarse_to_fine_interleaved(plan, corner_sizes[0], corner_sizes[1], corner_sizes[2],
          from_info, coarse_block, to_info, fine_block);
      }
    }
  }
}

/// Deinterleaves an even-length array of doubles.
/// @param size the length of both the output arrays.
/// @param in the input array.
/// @param rout the even-located elements.
/// @param iout the odd-located elements.
void deinterleave_real(const size_t size, const double *in, double *rout, double *iout)
{
  const double *in_e = (const double*) in;
  size_t i = 0;

#ifdef __SSE2__
  if ((((uintptr_t) in | (uintptr_t) rout | (uintptr_t) iout) & SSE_ALIGN_MASK) == 0)
  {
    for(; i + 2 <= size; i += 2)
    {
      __m128d first_in, second_in, first_out, second_out;
      first_in = _mm_load_pd(in_e + 2 * i);
      second_in = _mm_load_pd(in_e + 2 * i + 2);
      first_out = _mm_shuffle_pd(first_in, second_in, 0);
      second_out = _mm_shuffle_pd(first_in, second_in, 3);
      _mm_store_pd(rout + i, first_out);
      _mm_store_pd(iout + i, second_out);
    }
  }
#endif

  // This also handles the final element in the (size % 2 == 1) case.
  for(; i<size; ++i)
  {
    rout[i] = in_e[2 * i];
    iout[i] = in_e[2 * i + 1];
  }
}


/// Interleaves two arrays of doubles.
/// @param size the length of both the input arrays.
/// @param out the output array.
/// @param even the even-located elements.
/// @param odd the odd-located elements.
void interleave_real(size_t size, double *out, const double *even, const double *odd)
{
  size_t i = 0;

#ifdef __SSE2__
  if ((((uintptr_t) out | (uintptr_t) even | (uintptr_t) odd) & SSE_ALIGN_MASK) == 0)
  {
    for(; i + 2 <= size; i += 2)
    {
      __m128d even_vec, odd_vec, first_vec, second_vec;
      even_vec = _mm_load_pd((even + i));
      odd_vec = _mm_load_pd((odd + i));
      first_vec = _mm_shuffle_pd(even_vec, odd_vec, 0);
      second_vec = _mm_shuffle_pd(even_vec, odd_vec, 3);
      _mm_store_pd(out + i * 2, first_vec);
      _mm_store_pd(out + i * 2 + 2, second_vec);
    }
  }
#endif

  // This also handles the final element in the (size % 2 == 1) case.
  for(;i < size; ++i)
  {
    out[i * 2] = even[i];
    out[i * 2 + 1] = odd[i];
  }
}

void complex_to_product(const size_t size, const fftw_complex *in, double *out)
{
  double *in_e = (double*) in;

  for(size_t i=0; i<size; ++i)
    out[i] = in_e[2 * i] * in_e[2 * i + 1];
}

/// Time an interpolation plan for complex input data.
double time_interpolate_interleaved(interpolate_plan plan)
{
  ticks before, after;
  const size_t block_size = num_elements(plan);
  fftw_complex *const in = rs_alloc_complex(block_size);
  fftw_complex *const out = rs_alloc_complex(8 * block_size);

  memset(in, 0, block_size * sizeof(fftw_complex));
  memset(out, 0, 8 * block_size * sizeof(fftw_complex));

  before = getticks();
  interpolate_execute_interleaved(plan, in, out);
  after = getticks();

  rs_free(in);
  rs_free(out);

  return elapsed(after, before);
}


/// Time an interpolation plan for split-format input data.
double time_interpolate_split(interpolate_plan plan)
{
  ticks before, after;
  const size_t block_size = num_elements(plan);
  double *const in1 = rs_alloc_real(block_size);
  double *const in2 = rs_alloc_real(block_size);
  double *const out1 = rs_alloc_real(8 * block_size);
  double *const out2 = rs_alloc_real(8 * block_size);

  memset(in1, 0, block_size * sizeof(double));
  memset(in2, 0, block_size * sizeof(double));
  memset(out1, 0, 8 * block_size * sizeof(double));
  memset(out2, 0, 8 * block_size * sizeof(double));

  before = getticks();
  interpolate_execute_split(plan, in1, in2, out1, out2);
  after = getticks();

  rs_free(in1);
  rs_free(in2);
  rs_free(out1);
  rs_free(out2);

  return elapsed(after, before);
}

/// Time an interpolation for split-format data where the results are
/// pointwise multiplied.
double time_interpolate_split_product(interpolate_plan plan)
{
  ticks before, after;
  const size_t block_size = num_elements(plan);
  double *const in1 = rs_alloc_real(block_size);
  double *const in2 = rs_alloc_real(block_size);
  double *const out = rs_alloc_real(8 * block_size);

  memset(in1, 0, block_size * sizeof(double));
  memset(in2, 0, block_size * sizeof(double));
  memset(out, 0, 8 * block_size * sizeof(double));

  before = getticks();
  interpolate_execute_split_product(plan, in1, in2, out);
  after = getticks();

  rs_free(in1);
  rs_free(in2);
  rs_free(out);

  return elapsed(after, before);
}

/// Copy doubles from a smaller to larger region.
void copy_real(const block_info_t *from_info, const double *from,
  const block_info_t *to_info, double *to)
{
  const size_t n2 = from_info->dims[2];
  const size_t n1 = from_info->dims[1];
  const size_t n0 = from_info->dims[0];

  for(size_t i2 = 0; i2 < n2; ++i2)
  {
    for(size_t i1 = 0; i1 < n1; ++i1)
    {
      memcpy(to, from, sizeof(double) * n0);
      from += from_info->strides[1];
      to += to_info->strides[1];
    }

    from += from_info->strides[2] - n1 * from_info->strides[1];
    to += to_info->strides[2] - n1 * to_info->strides[1];
  }
}

void populate_strides_unpadded(block_info_t *info)
{
  info->strides[0] = 1;
  info->strides[1] = info->dims[0];
  info->strides[2] = info->dims[0] * info->dims[1];
}

void get_block_info_coarse(const interpolate_plan plan, block_info_t *info)
{
  info->dims[0] = plan_input_size(plan, 0);
  info->dims[1] = plan_input_size(plan, 1);
  info->dims[2] = plan_input_size(plan, 2);
  populate_strides_unpadded(info);
}

void get_block_info_fine(const interpolate_plan plan, block_info_t *info)
{
  info->dims[0] = plan_input_size(plan, 0) * 2;
  info->dims[1] = plan_input_size(plan, 1) * 2;
  info->dims[2] = plan_input_size(plan, 2) * 2;
  populate_strides_unpadded(info);
}

void get_block_info_real_recip_coarse(const interpolate_plan plan, block_info_t *info)
{
  info->dims[0] = plan_input_size(plan, 0) / 2 + 1;
  info->dims[1] = plan_input_size(plan, 1);
  info->dims[2] = plan_input_size(plan, 2);
  populate_strides_unpadded(info);
}

void get_block_info_real_recip_fine(const interpolate_plan plan, block_info_t *info)
{
  info->dims[0] = plan_input_size(plan, 0) + 1;
  info->dims[1] = plan_input_size(plan, 1) * 2;
  info->dims[2] = plan_input_size(plan, 2) * 2;
  populate_strides_unpadded(info);
}

void setup_threading(void)
{
#ifdef FFTW_IS_THREADED

#ifdef _OPENMP
#pragma omp critical
{
#endif
  static int fftw_initialised = 0;

  if (!fftw_initialised)
  {
    const int success = fftw_init_threads();
    assert(success);
    fftw_initialised = 1;
  }
#ifdef _OPENMP
}
#endif

#ifdef _OPENMP
#pragma omp critical
{
#ifdef FFTW_IS_MKL
  // FIXME: IMKL is not thread-safe for multiple threads executing the same
  // plan. We need to tell IMKL the number of threads using it explicitly. We
  // have no sane way to determine this value so we hard-wire it for now.
  fftw3_mkl.number_of_user_threads = 128;
#endif

  // Disable intra-FFT parallelisation for now.
  fftw_plan_with_nthreads(1);
}
#endif

#endif
}
