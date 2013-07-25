#ifndef COMMON_H
#define COMMON_J

#include <complex.h>
#include <stdint.h>
#include <fftw3.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

enum
{
  SSE_ALIGN = 1 << 4,
  SSE_ALIGN_MASK = SSE_ALIGN - 1
};

typedef enum
{
  INTERLEAVED,
  SPLIT,
  SPLIT_PRODUCT
} interpolation_t;

typedef struct
{
  int type;
  int dims[3];
  int strides[3];
} interpolate_properties_t;

typedef struct
{
  int dims[3];
  int strides[3];
} block_info_t;

static void pointwise_multiply_complex(size_t size, fftw_complex *a, const fftw_complex *b);
static void pointwise_multiply_real(size_t size, double *a, const double *b);
static void interleaved_to_split(const size_t size, const fftw_complex *in, double *rout, double *iout);
static void split_to_interleaved(const size_t size, const double *rin, const double *iin, fftw_complex *out);

void populate_properties(interpolate_properties_t *props, interpolation_t type, size_t n0, size_t n1, size_t n2);
void pad_coarse_to_fine_interleaved(interpolate_properties_t *props,
  const block_info_t *from_info, const fftw_complex *from,
  const block_info_t *to_info, fftw_complex *to,
  int positive_only);
void halve_nyquist_components(interpolate_properties_t *props, block_info_t *block_info, fftw_complex *coarse);

static inline void pointwise_multiply_complex(size_t size, fftw_complex *a, const fftw_complex *b)
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

static size_t num_elements(interpolate_properties_t *props)
{
  return props->dims[0] * props->dims[1] * props->dims[2];
}

static void get_block_info_coarse(interpolate_properties_t *props, block_info_t *info)
{
  info->dims[0] = props->dims[0];
  info->dims[1] = props->dims[1];
  info->dims[2] = props->dims[2];

  info->strides[0] = 1;
  info->strides[1] = info->dims[0];
  info->strides[2] = info->dims[0] * info->dims[1];
}

static void get_block_info_fine(interpolate_properties_t *props, block_info_t *info)
{
  info->dims[0] = props->dims[0] * 2;
  info->dims[1] = props->dims[1] * 2;
  info->dims[2] = props->dims[2] * 2;

  info->strides[0] = 1;
  info->strides[1] = info->dims[0];
  info->strides[2] = info->dims[0] * info->dims[1];
}

static void get_block_info_real_recip_coarse(interpolate_properties_t *props, block_info_t *info)
{
  info->dims[0] = props->dims[0] / 2 + 1;
  info->dims[1] = props->dims[1];
  info->dims[2] = props->dims[2];

  info->strides[0] = 1;
  info->strides[1] = info->dims[0];
  info->strides[2] = info->dims[0] * info->dims[1];
}

static void get_block_info_real_recip_fine(interpolate_properties_t *props, block_info_t *info)
{
  info->dims[0] = props->dims[0] + 1;
  info->dims[1] = props->dims[1] * 2;
  info->dims[2] = props->dims[2] * 2;

  info->strides[0] = 1;
  info->strides[1] = info->dims[0];
  info->strides[2] = info->dims[0] * info->dims[1];
}

static size_t num_elements_block(const block_info_t *block_info)
{
  return block_info->dims[0] * block_info->dims[1] * block_info->dims[2];
}

static inline size_t corner_size(const size_t n, const int negative)
{
  // In the even case, this will duplicate the Nyquist in both blocks
  return n / 2 + (negative == 0);
}

static inline void interleaved_to_split(const size_t size, const fftw_complex *in, double *rout, double *iout)
{
  const double *in_e = (const double*) in;
  for(size_t i=0; i<size; ++i)
  {
    rout[i] = in_e[2 * i];
    iout[i] = in_e[2 * i + 1];
  }
}

static void split_to_interleaved(const size_t size, const double *rin, const double *iin, fftw_complex *out)
{
  double *out_e = (double*) out;
  for(size_t i=0; i<size; ++i)
  {
    out_e[2 * i]  = rin[i];
    out_e[2 * i + 1] = iin[i];
  }
}

static inline void fftw_destroy_plan_maybe_null(fftw_plan plan)
{
  if (plan != NULL)
    fftw_destroy_plan(plan);
}

#endif
