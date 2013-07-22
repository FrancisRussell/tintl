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

static void pointwise_multiply_complex(int size, fftw_complex *a, const fftw_complex *b);
static void pointwise_multiply_real(int size, double *a, const double *b);

static inline void pointwise_multiply_complex(int size, fftw_complex *a, const fftw_complex *b)
{
#ifdef __SSE2__
  if ((((uintptr_t) b | (uintptr_t) a) & SSE_ALIGN_MASK) == 0)
  {
    // This *does* result in an observable performance improvement
    const __m128d neg = _mm_setr_pd(-1.0, 1.0);
    for(int i = 0; i<size; ++i)
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
    for(int i = 0; i < size; ++i)
      a[i] *= b[i];
#ifdef __SSE2__
  }
#endif
}

static inline void pointwise_multiply_real(int size, double *a, const double *b)
{
  int i = 0;

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

#endif
