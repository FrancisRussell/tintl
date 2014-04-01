#ifndef TINTL_COMMON_H
#define TINTL_COMMON_H

#include <complex.h>
#include <stdint.h>
#include "tintl/forward.h"
#include "tintl/interpolate.h"
#include <assert.h>
#include "tintl/timer.h"

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

static const unsigned interpolate_plan_magic_value = 0x1e7f21e2;

enum
{
  SSE_ALIGN = 1 << 4,
  SSE_ALIGN_MASK = SSE_ALIGN - 1
};

/// Enumeration of the different types of supported interpolation.
typedef enum
{
  INTERPOLATE_INTERLEAVED,
  INTERPOLATE_SPLIT,
  INTERPOLATE_SPLIT_PRODUCT
} interpolation_t;

/// Describes the dimensions and padding of a 3D block of data.
typedef struct
{
  int dims[3];
  int strides[3];
} block_info_t;

/// Data structure common to all different interpolation implementations.
///
/// Implementations populate this struct with a pointer to
/// implementation specific data and function pointers / that this data as
/// a parameter.

struct interpolate_plan_s
{
  /// Magic value
  unsigned magic;

  /// Type of interpolation
  interpolation_t type;

  /// Transform dimensions
  block_info_t input_size;

  /// Reference count
  int ref_cnt;

  // Timing
  time_point_t before;
  time_point_t after;

  const char *(*get_name)(const interpolate_plan plan);
  void (*set_flags)(interpolate_plan plan, int flags);
  void (*get_statistic_float)(const interpolate_plan plan, int statistic, int index, stat_type_t *type, double *value);
  void (*execute_interleaved)(interpolate_plan plan, rs_complex *in, rs_complex *out);
  void (*execute_split)(interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);
  void (*execute_split_product)(interpolate_plan plan, double *rin, double *iin, double *out);
  void (*print_timings)(const interpolate_plan plan);
  void (*destroy_detail)(interpolate_plan plan);
};

static void pointwise_multiply_complex(size_t size, rs_complex *a, const rs_complex *b);
static void pointwise_multiply_real(size_t size, double *a, const double *b);

void deinterleave_real(size_t size, const double *in, double *rout, double *iout);
void interleave_real(size_t size, double *out, const double *even, const double *odd);
void complex_to_product(const size_t size, const rs_complex *in, double *out);

void populate_strides_unpadded(block_info_t *info);
void get_block_info_coarse(const interpolate_plan plan, block_info_t *info);
void get_block_info_fine(const interpolate_plan plan, block_info_t *info);
void get_block_info_real_recip_coarse(const interpolate_plan plan, block_info_t *info);
void get_block_info_real_recip_fine(const interpolate_plan plan, block_info_t *info);

void populate_properties(interpolate_plan plan, interpolation_t type, size_t n0, size_t n1, size_t n2);
void pad_coarse_to_fine_interleaved(interpolate_plan plan,
  const block_info_t *from_info, const rs_complex *from,
  const block_info_t *to_info, rs_complex *to,
  int positive_only);
void copy_real(const block_info_t *from_info, const double *from,
  const block_info_t *to_info, double *to);
void halve_nyquist_components(interpolate_plan plan, block_info_t *block_info, rs_complex *coarse);

double time_interpolate_interleaved(interpolate_plan plan);
double time_interpolate_split(interpolate_plan plan);
double time_interpolate_split_product(interpolate_plan plan);

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

static inline size_t num_elements_block(const block_info_t *block_info)
{
  return block_info->dims[0] * block_info->dims[1] * block_info->dims[2];
}

static inline void validate_plan(const interpolate_plan plan)
{
  assert(plan != NULL && "Null pointer passed as interpolation plan handle.");
  assert(plan->magic == interpolate_plan_magic_value && "Corrupt or invalid plan detected.");
}

static inline interpolate_plan cast_to_parent(void *vplan)
{
  interpolate_plan plan = (interpolate_plan) vplan;
  validate_plan(plan);
  return plan;
}

static inline size_t num_elements(interpolate_plan plan)
{
  return num_elements_block(&plan->input_size);
}

static inline size_t corner_size(const size_t n, const int negative)
{
  // In the even case, this will duplicate the Nyquist in both blocks
  return n / 2 + (negative == 0);
}

static inline size_t plan_input_size(interpolate_plan plan, int dim)
{
  return plan->input_size.dims[dim];
}

#ifdef __cplusplus
}
#endif

#endif
