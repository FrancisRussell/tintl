#include "interpolate.h"
#include "phase_shift.h"
#include "timer.h"
#include "allocation.h"
#include "fftw_cycle.h"
#include <complex.h>
#include <stdint.h>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

static const int TIMING_ITERATIONS = 50;
static const double pi = 3.14159265358979323846;

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
  int stage[2][3];
  int interpolation;
  int strategy;
  int dims[3];
  int strides[3];
  int fine_dims[3];
  int fine_strides[3];

  fftw_plan naive_forward_interleaved;
  fftw_plan naive_backward_interleaved;

  fftw_plan dfts[3];
  fftw_plan dfts_staged[3];

  fftw_plan idfts[3];
  fftw_plan idfts_staged[3];

  fftw_complex *rotations[3];

  time_point_t before_expand2;
  time_point_t before_expand1;
  time_point_t before_expand0;
  time_point_t before_gather;
  time_point_t end;
} phase_shift_plan_s;

typedef phase_shift_plan_s *phase_shift_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */
static const char *get_name(const void *detail);
static void phase_shift_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out);
static void phase_shift_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout);
static void phase_shift_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out);
static void phase_shift_interpolate_print_timings(const void *detail);
static void phase_shift_interpolate_destroy_detail(void *detail);

static void plan_common(phase_shift_plan plan, int n0, int n1, int n2, int flags);

static void expand_dim0(phase_shift_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch);
static void expand_dim1(phase_shift_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch);
static void expand_dim2(phase_shift_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch);

static void expand_dim0_split(phase_shift_plan plan, double *in[2], double *out[2], fftw_complex *scratch);
static void expand_dim1_split(phase_shift_plan plan, double *in[2], double *out[2], fftw_complex *scratch);
static void expand_dim2_split(phase_shift_plan plan, double *in[2], double *out[2], fftw_complex *scratch);

static void gather_complex(int size, int stride, const fftw_complex *in, fftw_complex *out);
static void scatter_complex(int size, int stride, fftw_complex *out, const fftw_complex *in);

static void gather_split(int size, int stride, const double *rin, const double *iin, fftw_complex *out);
static void scatter_split(int size, int stride, double *rout, double *iout, const fftw_complex *in);

static void gather_blocks_real(phase_shift_plan plan, double *blocks[8], double *out);
static void gather_blocks_complex(phase_shift_plan plan, fftw_complex *blocks[8], fftw_complex *out);

static void interleave_complex(int size, fftw_complex *out, const fftw_complex *even, const fftw_complex *odd);
static void interleave_real(int size, double *out, const double *even, const double *odd);

static void pointwise_multiply_complex(int size, fftw_complex *a, const fftw_complex *b);
static void pointwise_multiply_real(int size, double *a, const double *b);

static void interpolate_split_common(const phase_shift_plan plan, double *blocks[8][2]);
static void build_rotation(int size, fftw_complex *out);
static int max_dimension(const phase_shift_plan plan);
static int round_align(int value);

static void pad_coarse_to_fine_interleaved(phase_shift_plan plan, const fftw_complex *from, fftw_complex *to);
static void halve_nyquist_components(phase_shift_plan plan, fftw_complex *coarse);

static const char *get_name(const void *detail)
{
  return "phase_shift";
}

static interpolate_plan allocate_plan(void)
{
  interpolate_plan holder = malloc(sizeof(interpolate_plan_s));
  holder->detail = malloc(sizeof(phase_shift_plan_s));
  holder->get_name = get_name;
  holder->execute_interleaved = phase_shift_interpolate_execute_interleaved;
  holder->execute_split = phase_shift_interpolate_execute_split;
  holder->execute_split_product = phase_shift_interpolate_execute_split_product;
  holder->print_timings = phase_shift_interpolate_print_timings;
  holder->destroy_detail = phase_shift_interpolate_destroy_detail;

  return holder;
}

static inline void stage_in_split(phase_shift_plan plan, int dim, double *rin, double *iin, fftw_complex *scratch)
{
  gather_split(plan->dims[dim], plan->strides[dim], rin, iin, scratch);
  fftw_execute_dft(plan->dfts_staged[dim], scratch, scratch);
}

static inline void stage_out_split(phase_shift_plan plan, int dim, double *rout, double *iout, fftw_complex *scratch)
{
  fftw_execute_dft(plan->idfts_staged[dim], scratch, scratch);
  scatter_split(plan->dims[dim], plan->strides[dim], rout, iout, scratch);
}

static inline void transform_in_split(phase_shift_plan plan, int dim, double *rin, double *iin, fftw_complex *scratch)
{
  fftw_execute_split_dft(plan->dfts[dim], rin, iin, (double*) scratch, ((double*) scratch) + 1);
}

static inline void transform_out_split(phase_shift_plan plan, int dim, double *rout, double *iout, fftw_complex *scratch)
{
  fftw_execute_split_dft(plan->idfts[dim], ((double*) scratch) + 1, (double*) scratch, iout, rout);
}

static inline void stage_in_interleaved(phase_shift_plan plan, int dim, fftw_complex *in, fftw_complex *scratch)
{
  gather_complex(plan->dims[dim], plan->strides[dim], in, scratch);
  fftw_execute_dft(plan->dfts_staged[dim], scratch, scratch);
}

static inline void stage_out_interleaved(phase_shift_plan plan, int dim, fftw_complex *out, fftw_complex *scratch)
{
  fftw_execute_dft(plan->idfts_staged[dim], scratch, scratch);
  scatter_complex(plan->dims[dim], plan->strides[dim], out, scratch);
}

static inline void transform_in_interleaved(phase_shift_plan plan, int dim, fftw_complex *in, fftw_complex *scratch)
{
  fftw_execute_dft(plan->dfts[dim], in, scratch);
}

static inline void transform_out_interleaved(phase_shift_plan plan, int dim, fftw_complex *out, fftw_complex *scratch)
{
  fftw_execute_dft(plan->idfts[dim], scratch, out);
}

static int round_align(const int value)
{
  const int remainder = value % SSE_ALIGN;
  return (remainder == 0 ? value : value + SSE_ALIGN - remainder);
}

static int max_dimension(const phase_shift_plan plan)
{
  int max_dim = 0;
  for(int dim=0; dim < 3; ++dim)
    max_dim = (max_dim < plan->dims[dim] ? plan->dims[dim] : max_dim);
  return max_dim;
}

static void plan_common(phase_shift_plan plan, int n0, int n1, int n2, int flags)
{
  flags |= FFTW_MEASURE;

  plan->dims[0] = n2;
  plan->dims[1] = n1;
  plan->dims[2] = n0;

  for(int dim = 0; dim < 3; ++dim)
    plan->fine_dims[dim] = plan->dims[dim] * 2;

  plan->strides[0] = 1;
  plan->strides[1] = n2;
  plan->strides[2] = n2 * n1;

  plan->fine_strides[0] = 1;
  plan->fine_strides[1] = n2 * 2;
  plan->fine_strides[2] = n2 * n1 * 4;

  for(int dim = 0; dim < 3; ++dim)
  {
    plan->rotations[dim] = rs_alloc_complex(plan->dims[dim]);
    build_rotation(plan->dims[dim], plan->rotations[dim]);
  }

  fftw_complex *const scratch = rs_alloc_complex(max_dimension(plan));

  for(int dim=0; dim < 3; ++dim)
  {
    plan->dfts_staged[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1,
      scratch, NULL, 1, 0,
      scratch, NULL, 1, 0,
      FFTW_FORWARD, flags);

    assert(plan->dfts_staged[dim] != NULL);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    plan->idfts_staged[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1,
      scratch, NULL, 1, 0,
      scratch, NULL, 1, 0,
      FFTW_BACKWARD, flags);

    assert(plan->idfts_staged[dim] != NULL);
  }

  rs_free(scratch);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  fftw_complex *const scratch_coarse = rs_alloc_complex(block_size);
  fftw_complex *const scratch_fine = rs_alloc_complex(8 * block_size);

  plan->naive_forward_interleaved = fftw_plan_dft(3, plan->dims, scratch_coarse, scratch_coarse, FFTW_FORWARD, flags);
  plan->naive_backward_interleaved = fftw_plan_dft(3, plan->fine_dims, scratch_fine, scratch_fine, FFTW_BACKWARD, flags);

  rs_free(scratch_coarse);
  rs_free(scratch_fine);
}

interpolate_plan interpolate_plan_3d_phase_shift_interleaved(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  phase_shift_plan plan = (phase_shift_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, n0, n1, n2, flags);
  plan->interpolation = INTERLEAVED;

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  fftw_complex *const data_in = rs_alloc_complex(block_size);
  fftw_complex *const data_out = rs_alloc_complex(block_size);
  fftw_complex *const scratch = rs_alloc_complex(max_dimension(plan));

  memset(data_in, 0, block_size * sizeof(fftw_complex));
  memset(data_out, 0, block_size * sizeof(fftw_complex));

  for(int dim=0; dim < 3; ++dim)
  {
    plan->dfts[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1,
      data_in, NULL, plan->strides[dim], 0,
      scratch, NULL, 1, 0,
      FFTW_FORWARD, flags);

    assert(plan->dfts[dim] != NULL);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    plan->idfts[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1,
      scratch, NULL, 1, 0,
      data_out, NULL, plan->strides[dim], 0,
      FFTW_BACKWARD, flags | FFTW_DESTROY_INPUT);

    assert(plan->idfts[dim] != NULL);
  }

  void (*transform_function[2])(phase_shift_plan, int, fftw_complex*, fftw_complex*) = {
    transform_in_interleaved,
    transform_out_interleaved
  };

  void (*staged_function[2])(phase_shift_plan, int, fftw_complex*, fftw_complex*) = {
    stage_in_interleaved,
    stage_out_interleaved
  };

  ticks before, after;

  for(int phase = 0; phase < 2; ++phase)
  {
    for(int dim=0; dim< 3; ++dim)
    {
      before = getticks();
      for(int repeat = 0; repeat < TIMING_ITERATIONS; ++repeat)
        transform_function[phase](plan, dim, data_in, scratch);
      after = getticks();
      const double transform_time = elapsed(after, before);

      before = getticks();
      for(int repeat = 0; repeat < TIMING_ITERATIONS; ++repeat)
        staged_function[phase](plan, dim, data_out, scratch);
      after = getticks();
      const double staged_time = elapsed(after, before);

      plan->stage[phase][dim] = (staged_time < transform_time);
    }
  }

  rs_free(scratch);
  rs_free(data_in);
  rs_free(data_out);
  return wrapper;
}

interpolate_plan interpolate_plan_3d_phase_shift_split(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = allocate_plan();
  phase_shift_plan plan = (phase_shift_plan) wrapper->detail;

  flags |= FFTW_MEASURE;
  plan_common(plan, n0, n1, n2, flags);
  plan->interpolation = SPLIT;

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  double *const real_scratch = rs_alloc_real(block_size);
  double *const imag_scratch = rs_alloc_real(block_size);
  fftw_complex *const scratch = rs_alloc_complex(max_dimension(plan));

  for(int dim=0; dim < 3; ++dim)
  {
    fftw_iodim dims;
    dims.n = plan->dims[dim];
    dims.is = plan->strides[dim];
    dims.os = 2;

    plan->dfts[dim] = fftw_plan_guru_split_dft(1, &dims, 0, NULL,
      real_scratch, imag_scratch,
      (double*) scratch, ((double*) scratch)+1,
      flags);

    assert(plan->dfts[dim] != NULL);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    fftw_iodim dims;
    dims.n = plan->dims[dim];
    dims.is = 2;
    dims.os = plan->strides[dim];

    plan->idfts[dim] = fftw_plan_guru_split_dft(1, &dims, 0, NULL,
      ((double*) scratch) + 1, ((double*) scratch),
      real_scratch, imag_scratch,
      flags | FFTW_DESTROY_INPUT);

    assert(plan->idfts[dim] != NULL);
  }

  double *const real_scratch_2 = rs_alloc_real(block_size);
  double *const imag_scratch_2 = rs_alloc_real(block_size);

  memset(real_scratch_2, 0, block_size * sizeof(double));
  memset(imag_scratch_2, 0, block_size * sizeof(double));

  void (*transform_function[2])(phase_shift_plan, int, double*, double*, fftw_complex*) = {
    transform_in_split,
    transform_out_split
  };

  void (*staged_function[2])(phase_shift_plan, int, double*, double*, fftw_complex*) = {
    stage_in_split,
    stage_out_split
  };

  ticks before, after;

  for(int phase = 0; phase < 2; ++phase)
  {
    for(int dim=0; dim< 3; ++dim)
    {
      before = getticks();
      for(int repeat = 0; repeat < TIMING_ITERATIONS; ++repeat)
        transform_function[phase](plan, dim, real_scratch, imag_scratch, scratch);
      after = getticks();
      const double transform_time = elapsed(after, before);

      before = getticks();
      for(int repeat = 0; repeat < TIMING_ITERATIONS; ++repeat)
        staged_function[phase](plan, dim, real_scratch_2, imag_scratch_2, scratch);
      after = getticks();
      const double staged_time = elapsed(after, before);

      plan->stage[phase][dim] = (staged_time < transform_time);
    }
  }

  rs_free(scratch);
  rs_free(imag_scratch);
  rs_free(imag_scratch_2);
  rs_free(real_scratch);
  rs_free(real_scratch_2);
  return wrapper;
}

interpolate_plan plan_interpolate_3d_split_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan wrapper = interpolate_plan_3d_phase_shift_split(n0, n1, n2, flags);
  ((phase_shift_plan) wrapper->detail)->interpolation = SPLIT_PRODUCT;
  return wrapper;
}

static void phase_shift_interpolate_destroy_detail(void *detail)
{
  phase_shift_plan plan = (phase_shift_plan) detail;

  fftw_destroy_plan(plan->naive_forward_interleaved);
  fftw_destroy_plan(plan->naive_backward_interleaved);

  for(int dim = 0; dim < 3; ++dim)
  {
    rs_free(plan->rotations[dim]);

    if (plan->dfts[dim] != NULL)
      fftw_destroy_plan(plan->dfts[dim]);

    if (plan->dfts_staged[dim] != NULL)
      fftw_destroy_plan(plan->dfts_staged[dim]);

    if (plan->idfts[dim] != NULL)
      fftw_destroy_plan(plan->idfts[dim]);

    if (plan->idfts_staged[dim] != NULL)
      fftw_destroy_plan(plan->idfts_staged[dim]);
  }

  free(plan);
}

static void build_rotation(int size, fftw_complex *out)
{
  const double theta_base = pi/size;

  for(int freq = 0; freq < size; ++freq)
  {
    if (size % 2 == 0 && freq == size / 2)
    {
      out[freq] = 0.0;
    }
    else
    {
      double theta;
      if (freq < 1 + size / 2)
        theta = theta_base * freq;
      else
        theta = pi + (theta_base * freq);

      out[freq] = cos(theta) / size + I * sin(theta) / size;
    }
  }
}

static void interleave_complex(int size, fftw_complex *out, const fftw_complex *even, const fftw_complex *odd)
{
#ifdef __SSE2__
  if ((((uintptr_t) out | (uintptr_t) even | (uintptr_t) odd) & SSE_ALIGN_MASK) == 0)
  {
    // This does not result in any observable performance improvement
    for(int i = 0; i < size; ++i)
    {
      __m128d even_vec = _mm_load_pd((const double*)(even + i));
      __m128d odd_vec = _mm_load_pd((const double*)(odd + i));
      _mm_store_pd((double*)(out + i*2), even_vec);
      _mm_store_pd((double*)(out + i*2 + 1), odd_vec);
    }
  }
  else
  {
#endif
    for(int i = 0; i < size; ++i)
    {
      out[i*2] = even[i];
      out[i*2 + 1] = odd[i];
    }
#ifdef __SSE2__
  }
#endif
}

static void interleave_real(int size, double *out, const double *even, const double *odd)
{
  int i = 0;

#ifdef __SSE2__
  if ((((uintptr_t) out | (uintptr_t) even | (uintptr_t) odd) & SSE_ALIGN_MASK) == 0)
  {
    for(; i + (SSE_ALIGN / sizeof(double)) <= size; i += (SSE_ALIGN / sizeof(double)))
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
    out[i*2] = even[i];
    out[i*2 + 1] = odd[i];
  }
}

static void pointwise_multiply_complex(int size, fftw_complex *a, const fftw_complex *b)
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

static void pointwise_multiply_real(int size, double *a, const double *b)
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

static void gather_complex(int size, int stride, const fftw_complex *in, fftw_complex *out)
{
  for(int i=0; i<size; ++i)
    out[i] = in[i*stride];
}

static void scatter_complex(int size, int stride, fftw_complex *out, const fftw_complex *in)
{
  for(int i=0; i<size; ++i)
    out[i*stride] = in[i];
}

static void gather_split(int size, int stride, const double *rin, const double *iin, fftw_complex *cout)
{
  double *const out = (double*) cout;

  for(int i=0; i < size; ++i)
  {
    out[i * 2] = rin[i * stride];
    out[i * 2 + 1] = iin[i * stride];
  }
}

static void scatter_split(int size, int stride, double *rout, double *iout, const fftw_complex *cin)
{
  double *const in = (double*) cin;

  for(int i=0; i < size; ++i)
  {
    rout[i * stride] = in[2 * i];
    iout[i * stride] = in[2 * i + 1];
  }
}

static void gather_blocks_complex(phase_shift_plan plan, fftw_complex *blocks[8], fftw_complex *out)
{
  for(int i2=0; i2 < plan->dims[2] * 2; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1] * 2; ++i1)
    {
      const int in_offset = (i2/2) * plan->strides[2] + (i1/2) * plan->strides[1];
      const fftw_complex *even = &blocks[(i2 % 2) + (i1 % 2) * 2][in_offset];
      const fftw_complex *odd = &blocks[(i2 % 2) + (i1 % 2) * 2 + 4][in_offset];
      fftw_complex *row_out = &out[i2 * plan->strides[2] * 4 + i1 * plan->strides[1] * 2];
      interleave_complex(plan->dims[0], row_out, even, odd);
    }
  }
}

static void gather_blocks_real(phase_shift_plan plan, double *blocks[8], double *out)
{
  for(int i2=0; i2 < plan->dims[2] * 2; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1] * 2; ++i1)
    {
      const int in_offset = (i2/2) * plan->strides[2] + (i1/2) * plan->strides[1];
      const double *even = &blocks[(i2 % 2) + (i1 % 2) * 2][in_offset];
      const double *odd = &blocks[(i2 % 2) + (i1 % 2) * 2 + 4][in_offset];
      double *row_out = &out[i2 * plan->strides[2] * 4 + i1 * plan->strides[1] * 2];
      interleave_real(plan->dims[0], row_out, even, odd);
    }
  }
}

static void phase_shift_interpolate_execute_interleaved(const void *detail, fftw_complex *in, fftw_complex *out)
{
  phase_shift_plan plan = (phase_shift_plan) detail;
  assert(INTERLEAVED == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];

  if (1)
  {
    fftw_complex *const block_data = rs_alloc_complex(7 * block_size);
    fftw_complex *blocks[8];
    blocks[0] = in;

    for(int block = 1; block < 8; ++block)
      blocks[block] = block_data + (block - 1) * block_size;

    const int max_dim = max_dimension(plan);
    fftw_complex *const scratch = rs_alloc_complex(max_dim);

    time_point_save(&plan->before_expand2);
    expand_dim2(plan, blocks[0], blocks[1], scratch);

    time_point_save(&plan->before_expand1);
    for(int n = 0; n < 2; ++n)
      expand_dim1(plan, blocks[n], blocks[n + 2], scratch);

    time_point_save(&plan->before_expand0);
    for(int n = 0; n < 4; ++n)
      expand_dim0(plan, blocks[n], blocks[n + 4], scratch);

    time_point_save(&plan->before_gather);
    gather_blocks_complex(plan, blocks, out);
    time_point_save(&plan->end);

    rs_free(scratch);
    rs_free(block_data);
  }
  else
  {
    fftw_complex *const input_copy = rs_alloc_complex(block_size);
    memcpy(input_copy, in, sizeof(fftw_complex) * block_size);

    fftw_execute_dft(plan->naive_forward_interleaved, input_copy, input_copy);
    halve_nyquist_components(plan, input_copy);
    pad_coarse_to_fine_interleaved(plan, input_copy, out);
    fftw_execute_dft(plan->naive_backward_interleaved, out, out);

    rs_free(input_copy);
  }
}

static void interpolate_split_common(const phase_shift_plan plan, double *blocks[8][2])
{
  const int max_dim = max_dimension(plan);
  fftw_complex *const scratch = rs_alloc_complex(max_dim);

  time_point_save(&plan->before_expand2);
  expand_dim2_split(plan, blocks[0], blocks[1], scratch);

  time_point_save(&plan->before_expand1);
  for(int n = 0; n < 2; ++n)
    expand_dim1_split(plan, blocks[n], blocks[n + 2], scratch);

  time_point_save(&plan->before_expand0);
  for(int n = 0; n < 4; ++n)
    expand_dim0_split(plan, blocks[n], blocks[n + 4], scratch);

  rs_free(scratch);
}

static void phase_shift_interpolate_execute_split(const void *detail, double *rin, double *iin, double *rout, double *iout)
{
  phase_shift_plan plan = (phase_shift_plan) detail;
  assert(SPLIT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  const int rounded_block_size = round_align(block_size);
  double *const block_data = rs_alloc_real(2 * 7 * rounded_block_size);
  double *blocks[8][2];

  blocks[0][0] = rin;
  blocks[0][1] = iin;

  for(int block = 1; block < 8; ++block)
  {
    blocks[block][0] = block_data + (2 * (block - 1)) * rounded_block_size;
    blocks[block][1] = block_data + (2 * (block - 1) + 1) * rounded_block_size;
  }

  interpolate_split_common(plan, blocks);

  time_point_save(&plan->before_gather);

  double *real_blocks[8];
  double *imag_blocks[8];
  for(int block = 0; block < 8; ++block)
  {
    real_blocks[block] = blocks[block][0];
    imag_blocks[block] = blocks[block][1];
  }

  gather_blocks_real(plan, real_blocks, rout);
  gather_blocks_real(plan, imag_blocks, iout);
  time_point_save(&plan->end);

  rs_free(block_data);
}

void phase_shift_interpolate_execute_split_product(const void *detail, double *rin, double *iin, double *out)
{
  phase_shift_plan plan = (phase_shift_plan) detail;
  assert(SPLIT_PRODUCT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  const int rounded_block_size = round_align(block_size);
  double *const block_data = rs_alloc_real(2 * 7 * rounded_block_size);
  double *blocks[8][2];

  blocks[0][0] = rin;
  blocks[0][1] = iin;

  for(int block = 1; block < 8; ++block)
  {
    blocks[block][0] = block_data + (2 * (block - 1)) * rounded_block_size;
    blocks[block][1] = block_data + (2 * (block - 1) + 1) * rounded_block_size;
  }

  interpolate_split_common(plan, blocks);

  time_point_save(&plan->before_gather);

  for(int block = 0; block < 8; ++block)
    pointwise_multiply_real(block_size, blocks[block][0], blocks[block][1]);

  double *result_blocks[8];
  for(int block = 0; block < 8; ++block)
    result_blocks[block] = blocks[block][0];

  gather_blocks_real(plan, result_blocks, out);
  time_point_save(&plan->end);

  rs_free(block_data);
}

void phase_shift_interpolate_print_timings(const void *detail)
{
  phase_shift_plan plan = (phase_shift_plan) detail;
  printf("Expand2: %f\n", time_point_delta(&plan->before_expand2, &plan->before_expand1));
  printf("Expand1: %f\n", time_point_delta(&plan->before_expand1, &plan->before_expand0));
  printf("Expand0: %f\n", time_point_delta(&plan->before_expand0, &plan->before_gather));
  printf("Gather: %f\n", time_point_delta(&plan->before_gather, &plan->end));
}

static void expand_dim0(phase_shift_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch)
{
  static const int dim = 0;
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1]; ++i1)
    {
      const size_t offset = i1*plan->strides[1] + i2*plan->strides[2];

      if (plan->stage[0][dim])
        stage_in_interleaved(plan, dim, in + offset, scratch);
      else
        transform_in_interleaved(plan, dim, in + offset, scratch);

        pointwise_multiply_complex(plan->dims[dim], scratch, plan->rotations[dim]);

      if (plan->stage[1][dim])
        stage_out_interleaved(plan, dim, out + offset, scratch);
      else
        transform_out_interleaved(plan, dim, out + offset, scratch);
    }
  }
}

static void expand_dim0_split(phase_shift_plan plan, double *in[2], double *out[2], fftw_complex *scratch)
{
  static const int dim = 0;
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1]; ++i1)
    {
      const size_t offset = i1*plan->strides[1] + i2*plan->strides[2];

      if (plan->stage[0][dim])
        stage_in_split(plan, dim, in[0] + offset, in[1] + offset, scratch);
      else
        transform_in_split(plan, dim, in[0] + offset, in[1] + offset, scratch);

        pointwise_multiply_complex(plan->dims[dim], scratch, plan->rotations[dim]);

      if (plan->stage[1][dim])
        stage_out_split(plan, dim, out[0] + offset, out[1] + offset, scratch);
      else
        transform_out_split(plan, dim, out[0] + offset, out[1] + offset, scratch);
    }
  }
}


static void expand_dim1(phase_shift_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch)
{
  static const int dim = 1;
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i0 + i2*plan->strides[2];

      if (plan->stage[0][dim])
        stage_in_interleaved(plan, dim, in + offset, scratch);
      else
        transform_in_interleaved(plan, dim, in + offset, scratch);

        pointwise_multiply_complex(plan->dims[dim], scratch, plan->rotations[dim]);

      if (plan->stage[1][dim])
        stage_out_interleaved(plan, dim, out + offset, scratch);
      else
        transform_out_interleaved(plan, dim, out + offset, scratch);
    }
  }
}

static void expand_dim1_split(phase_shift_plan plan, double *in[2], double *out[2], fftw_complex *scratch)
{
  static const int dim = 1;
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i0 + i2*plan->strides[2];

      if (plan->stage[0][dim])
        stage_in_split(plan, dim, in[0] + offset, in[1] + offset, scratch);
      else
        transform_in_split(plan, dim, in[0] + offset, in[1] + offset, scratch);

        pointwise_multiply_complex(plan->dims[dim], scratch, plan->rotations[dim]);

      if (plan->stage[1][dim])
        stage_out_split(plan, dim, out[0] + offset, out[1] + offset, scratch);
      else
        transform_out_split(plan, dim, out[0] + offset, out[1] + offset, scratch);
    }
  }
}

static void expand_dim2(phase_shift_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch)
{
  static const int dim = 2;
  for(int i1=0; i1 < plan->dims[1]; ++i1)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i1*plan->strides[1] + i0;

      if (plan->stage[0][dim])
        stage_in_interleaved(plan, dim, in + offset, scratch);
      else
        transform_in_interleaved(plan, dim, in + offset, scratch);

        pointwise_multiply_complex(plan->dims[dim], scratch, plan->rotations[dim]);

      if (plan->stage[1][dim])
        stage_out_interleaved(plan, dim, out + offset, scratch);
      else
        transform_out_interleaved(plan, dim, out + offset, scratch);
    }
  }
}

static void expand_dim2_split(phase_shift_plan plan, double *in[2], double *out[2], fftw_complex *scratch)
{
  static const int dim = 2;
  for(int i1=0; i1 < plan->dims[1]; ++i1)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i1*plan->strides[1] + i0;

      if (plan->stage[0][dim])
        stage_in_split(plan, dim, in[0] + offset, in[1] + offset, scratch);
      else
        transform_in_split(plan, dim, in[0] + offset, in[1] + offset, scratch);

        pointwise_multiply_complex(plan->dims[dim], scratch, plan->rotations[dim]);

      if (plan->stage[1][dim])
        stage_out_split(plan, dim, out[0] + offset, out[1] + offset, scratch);
      else
        transform_out_split(plan, dim, out[0] + offset, out[1] + offset, scratch);
    }
  }
}

static void block_copy_coarse_to_fine_interleaved(phase_shift_plan plan, int n0, int n1, int n2, const fftw_complex *from, fftw_complex *to)
{
  assert(n0 <= plan->dims[0]);
  assert(n1 <= plan->dims[1]);
  assert(n2 <= plan->dims[2]);

  const double scale_factor = 1.0 / (plan->dims[0] * plan->dims[1] * plan->dims[2]);

  for(int i2=0; i2 < n2; ++i2)
  {
    for(int i1=0; i1 < n1; ++i1)
    {
      for(int i0=0; i0 < n0; ++i0)
      {
        to[i0] = from[i0] * scale_factor;
      }
        from += plan->strides[1];
      to += plan->fine_strides[1];
    }

    from += plan->strides[2] - n1 * plan->strides[1];
    to += plan->fine_strides[2] - n1 * plan->fine_strides[1];
  }
}

void halve_nyquist_components(phase_shift_plan plan, fftw_complex *coarse)
{
  const int n2 = plan->dims[2];
  const int n1 = plan->dims[1];
  const int n0 = plan->dims[0];

  const int s2 = plan->strides[2];
  const int s1 = plan->strides[1];

  if (n2 % 2 == 0)
    for(int i1 = 0; i1 < n1; ++i1)
      for(int i0 = 0; i0 < n0; ++i0)
        coarse[s2 * (n2 / 2) +  s1 * i1 + i0] *= 0.5;

  if (n1 % 2 == 0)
    for(int i2 = 0; i2 < n2; ++i2)
      for(int i0 = 0; i0 < n0; ++i0)
        coarse[s2 * i2 +  s1 * (n1 / 2) + i0] *= 0.5;

  if (n0 % 2 == 0)
    for(int i2 = 0; i2 < n2; ++i2)
      for(int i1 = 0; i1 < n1; ++i1)
        coarse[s2 * i2 +  s1 * i1 + (n0 / 2)] *= 0.5;
}

static int corner_size(const int n, const int negative)
{
  // In the even case, this will duplicate the Nyquist in both blocks
  return n / 2 + (negative == 0);
}

static void pad_coarse_to_fine_interleaved(phase_shift_plan plan, const fftw_complex *from, fftw_complex *to)
{

  const int coarse_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  memset(to, 0, 8 * coarse_size);

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
          corner_sizes[dim] = corner_size(plan->dims[dim], corner_flags[dim]);
          const int coarse_index = (corner_flags[dim] == 0) ? 0 : plan->dims[dim] - corner_sizes[dim];
          const int fine_index = (corner_flags[dim] == 0) ? 0 : plan->fine_dims[dim] - corner_sizes[dim];

          coarse_block += plan->strides[dim] * coarse_index;
          fine_block += plan->fine_strides[dim] * fine_index;
        }

        block_copy_coarse_to_fine_interleaved(plan, corner_sizes[0], corner_sizes[1], corner_sizes[2], coarse_block, fine_block);
      }
    }
  }
}
