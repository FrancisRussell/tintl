#ifndef COMMON_H
#define COMMON_H

#include <complex.h>
#include <stdint.h>
#include <forward.h>
#include <interpolate.h>

#ifdef __CUDACC__
#undef __SSE2__
#endif

#ifdef __SSE2__
#include <emmintrin.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

enum
{
  SSE_ALIGN = 1 << 4,
  SSE_ALIGN_MASK = SSE_ALIGN - 1
};

/// Enumeration of the different types of supported interpolation.
typedef enum
{
  INTERLEAVED,
  SPLIT,
  SPLIT_PRODUCT
} interpolation_t;

/// Describes an interpolation.
typedef struct
{
  int type;
  int dims[3];
  int strides[3];
} interpolate_properties_t;

/// Describes the dimensions and padding of a 3D block of data.
typedef struct
{
  int dims[3];
  int strides[3];
} block_info_t;

static void pointwise_multiply_complex(size_t size, rs_complex *a, const rs_complex *b);
static void pointwise_multiply_real(size_t size, double *a, const double *b);

void deinterleave_real(size_t size, const double *in, double *rout, double *iout);
void interleave_real(size_t size, double *out, const double *even, const double *odd);
void complex_to_product(const size_t size, const rs_complex *in, double *out);

void populate_strides_unpadded(block_info_t *info);
void get_block_info_coarse(const interpolate_properties_t *props, block_info_t *info);
void get_block_info_fine(const interpolate_properties_t *props, block_info_t *info);
void get_block_info_real_recip_coarse(const interpolate_properties_t *props, block_info_t *info);
void get_block_info_real_recip_fine(const interpolate_properties_t *props, block_info_t *info);

void populate_properties(interpolate_properties_t *props, interpolation_t type, size_t n0, size_t n1, size_t n2);
void pad_coarse_to_fine_interleaved(interpolate_properties_t *props,
  const block_info_t *from_info, const rs_complex *from,
  const block_info_t *to_info, rs_complex *to,
  int positive_only);
void copy_real(const block_info_t *from_info, const double *from,
  const block_info_t *to_info, double *to);
void halve_nyquist_components(interpolate_properties_t *props, block_info_t *block_info, rs_complex *coarse);

double time_interpolate_interleaved(interpolate_plan plan, const int *dims);
double time_interpolate_split(interpolate_plan plan, const int *dims);
double time_interpolate_split_product(interpolate_plan plan, const int *dims);

void setup_threading(void);

static inline void pointwise_multiply_complex(size_t size, rs_complex *a, const rs_complex *b)
{
#ifdef __SSE2__
  if ((((uintptr_t) b | (uintptr_t) a) & SSE_ALIGN_MASK) == 0)
  {
    // This *does* result in an observable performance improvement
    const __m128d neg = _mm_setr_pd(-1.0, 1.0);
    for(size_t i = 0; i<size; ++i)
    {
      __m128d a_vec, a_imag, a_real, b_vec, res;
      a_vec = _mm_load_pd((const double*)(a + i));
      b_vec = _mm_load_pd((const double*)(b + i));
      a_imag = _mm_shuffle_pd(a_vec, a_vec, 3);
      a_real = _mm_shuffle_pd(a_vec, a_vec, 0);
      res = _mm_mul_pd(b_vec, a_real);
      b_vec = _mm_shuffle_pd(b_vec, b_vec, 1);
      b_vec = _mm_mul_pd(b_vec, neg);
      b_vec = _mm_mul_pd(b_vec, a_imag);
      res = _mm_add_pd(res, b_vec);
      _mm_store_pd((double*)(a + i), res);
    }
  }
  else
  {
#endif
    for(size_t i = 0; i < size; ++i)
      a[i] *= b[i];
#ifdef __SSE2__
  }
#endif
}

static inline void pointwise_multiply_real(size_t size, double *a, const double *b)
{
  size_t i = 0;

#ifdef __SSE2__
  if ((((uintptr_t) a | (uintptr_t) b) & SSE_ALIGN_MASK) == 0)
  {
    for(; i + (SSE_ALIGN / sizeof(double)) <= size; i += (SSE_ALIGN / sizeof(double)))
    {
      __m128d a_vec, b_vec, res;
      a_vec = _mm_load_pd(a + i);
      b_vec = _mm_load_pd(b + i);
      res = _mm_mul_pd(a_vec, b_vec);
      _mm_store_pd((a + i), res);
    }
  }
#endif

  // This also handles the final element in the (size % 2 == 1) case.
  for(; i < size; ++i)
    a[i] *= b[i];
}

static inline size_t num_elements(interpolate_properties_t *props)
{
  return props->dims[0] * props->dims[1] * props->dims[2];
}

static inline size_t num_elements_block(const block_info_t *block_info)
{
  return block_info->dims[0] * block_info->dims[1] * block_info->dims[2];
}

static inline size_t corner_size(const size_t n, const int negative)
{
  // In the even case, this will duplicate the Nyquist in both blocks
  return n / 2 + (negative == 0);
}

#ifdef __cplusplus
}
#endif

#endif
