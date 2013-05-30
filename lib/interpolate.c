#include "interpolate.h"
#include "timer.h"
#include <complex.h>
#include <stdint.h>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

static const double pi = 3.14159265358979323846;

typedef enum
{
  INTERLEAVED,
  SPLIT,
  SPLIT_PRODUCT
} interpolation_t;

static void expand_dim0(interpolate_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch);
static void expand_dim1(interpolate_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch);
static void expand_dim2(interpolate_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch);

static void expand_dim0_split(interpolate_plan plan, double *in[2], double *out[2], fftw_complex *scratch);
static void expand_dim1_split(interpolate_plan plan, double *in[2], double *out[2], fftw_complex *scratch);
static void expand_dim2_split(interpolate_plan plan, double *in[2], double *out[2], fftw_complex *scratch);

static void gather_blocks_real(interpolate_plan plan, double *blocks[8], double *out);
static void gather_blocks_complex(interpolate_plan plan, fftw_complex *blocks[8], fftw_complex *out);

static void interleave_complex(int size, fftw_complex *out, const fftw_complex *even, const fftw_complex *odd);
static void interleave_real(int size, double *out, const double *even, const double *odd);

static void pointwise_multiply_complex(int size, fftw_complex *a, const fftw_complex *b);
static void pointwise_multiply_real(int size, double *a, const double *b);

static void interpolate_split_common(const interpolate_plan plan, double *blocks[8][2]);
static void build_rotation(int size, fftw_complex *out);
static int max_dimension(const interpolate_plan plan);
static int round16(int value);

static int round16(const int value)
{
  const int remainder = value % 16;
  return (remainder == 0 ? value : value + 16 - remainder);
}

static int max_dimension(const interpolate_plan plan)
{
  int max_dim = 0;
  for(int dim=0; dim < 3; ++dim)
    max_dim = (max_dim < plan->dims[dim] ? plan->dims[dim] : max_dim);
  return max_dim;
}

interpolate_plan plan_interpolate_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int flags)
{
  flags |= FFTW_MEASURE;
  interpolate_plan_s *const plan = malloc(sizeof(interpolate_plan_s));

  plan->interpolation = INTERLEAVED;
  plan->dims[0] = n2;
  plan->dims[1] = n1;
  plan->dims[2] = n0;

  plan->strides[0] = 1;
  plan->strides[1] = n2;
  plan->strides[2] = n2 * n1;

  for(int dim = 0; dim < 3; ++dim)
  {
    plan->rotations[dim] = fftw_alloc_complex(plan->dims[dim]);
    build_rotation(plan->dims[dim], plan->rotations[dim]);
  }

  fftw_complex *const scratch = fftw_alloc_complex(max_dimension(plan));

  for(int dim=0; dim < 3; ++dim)
  {
    plan->dfts[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1,
      in, NULL, plan->strides[dim], 0,
      scratch, NULL, 1, 0,
      FFTW_FORWARD, flags);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    plan->idfts[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1,
      scratch, NULL, 1, 0,
      out, NULL, plan->strides[dim], 0,
      FFTW_BACKWARD, flags | FFTW_DESTROY_INPUT);
  }

  fftw_free(scratch);
  return plan;
}

interpolate_plan plan_interpolate_3d_split(int n0, int n1, int n2, int flags)
{
  flags |= FFTW_MEASURE;
  interpolate_plan_s *const plan = malloc(sizeof(interpolate_plan_s));

  plan->interpolation = SPLIT;
  plan->dims[0] = n2;
  plan->dims[1] = n1;
  plan->dims[2] = n0;

  plan->strides[0] = 1;
  plan->strides[1] = n2;
  plan->strides[2] = n2 * n1;

  for(int dim = 0; dim < 3; ++dim)
  {
    plan->rotations[dim] = fftw_alloc_complex(plan->dims[dim]);
    build_rotation(plan->dims[dim], plan->rotations[dim]);
  }

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  double *const realScratch = fftw_alloc_real(block_size);
  double *const imagScratch = fftw_alloc_real(block_size);
  fftw_complex *const scratch = fftw_alloc_complex(max_dimension(plan));

  for(int dim=0; dim < 3; ++dim)
  {
    fftw_iodim dims;
    dims.n = plan->dims[dim];
    dims.is = plan->strides[dim];
    dims.os = 2;

    plan->dfts[dim] = fftw_plan_guru_split_dft(1, &dims, 0, NULL,
      realScratch, imagScratch,
      (double*) scratch, ((double*) scratch)+1,
      flags);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    fftw_iodim dims;
    dims.n = plan->dims[dim];
    dims.is = 2;
    dims.os = plan->strides[dim];

    plan->idfts[dim] = fftw_plan_guru_split_dft(1, &dims, 0, NULL,
      ((double*) scratch) + 1, ((double*) scratch),
      realScratch, imagScratch,
      flags | FFTW_DESTROY_INPUT);
  }

  fftw_free(scratch);
  fftw_free(imagScratch);
  fftw_free(realScratch);
  return plan;
}

interpolate_plan plan_interpolate_3d_split_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan plan = plan_interpolate_3d_split(n0, n1, n2, flags);
  plan->interpolation = SPLIT_PRODUCT;
  return plan;
}

void interpolate_destroy_plan(interpolate_plan plan)
{
  for(int dim = 0; dim < 3; ++dim)
    fftw_free(plan->rotations[dim]);

  for(int dim = 0; dim < 3; ++dim)
    fftw_destroy_plan(plan->dfts[dim]);

  for(int dim = 0; dim < 3; ++dim)
    fftw_destroy_plan(plan->idfts[dim]);

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
  // This does not result in any observable performance improvement
  for(int i = 0; i < size; ++i)
  {
    __m128d even_vec = _mm_load_pd((const double*)(even + i));
    __m128d odd_vec = _mm_load_pd((const double*)(odd + i));
    _mm_store_pd((double*)(out + i*2), even_vec);
    _mm_store_pd((double*)(out + i*2 + 1), odd_vec);
  }
#else
  for(int i = 0; i < size; ++i)
  {
    out[i*2] = even[i];
    out[i*2 + 1] = odd[i];
  }
#endif
}

static void interleave_real(int size, double *out, const double *even, const double *odd)
{
  for(int i = 0; i < size; ++i)
  {
    out[i*2] = even[i];
    out[i*2 + 1] = odd[i];
  }
}

static void pointwise_multiply_complex(int size, fftw_complex *a, const fftw_complex *b)
{
#ifdef __SSE2__
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
#else
  for(size_t i = 0; i < size; ++i)
    a[i] *= b[i];
#endif
}

static void pointwise_multiply_real(int size, double *a, const double *b)
{
  for(size_t i = 0; i < size; ++i)
    a[i] *= b[i];
}

static void gather_blocks_complex(interpolate_plan plan, fftw_complex *blocks[8], fftw_complex *out)
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

static void gather_blocks_real(interpolate_plan plan, double *blocks[8], double *out)
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

void interpolate_execute(const interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  assert(INTERLEAVED == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  fftw_complex *const block_data = fftw_alloc_complex(7 * block_size);
  fftw_complex *blocks[8];
  blocks[0] = in;

  for(int block = 1; block < 8; ++block)
    blocks[block] = block_data + (block - 1) * block_size;

  const int max_dim = max_dimension(plan);
  fftw_complex *const scratch = fftw_alloc_complex(max_dim);

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

  fftw_free(scratch);
  fftw_free(block_data);
}

static void interpolate_split_common(const interpolate_plan plan, double *blocks[8][2])
{
  const int max_dim = max_dimension(plan);
  fftw_complex *const scratch = fftw_alloc_complex(max_dim);

  time_point_save(&plan->before_expand2);
  expand_dim2_split(plan, blocks[0], blocks[1], scratch);

  time_point_save(&plan->before_expand1);
  for(int n = 0; n < 2; ++n)
    expand_dim1_split(plan, blocks[n], blocks[n + 2], scratch);

  time_point_save(&plan->before_expand0);
  for(int n = 0; n < 4; ++n)
    expand_dim0_split(plan, blocks[n], blocks[n + 4], scratch);

  fftw_free(scratch);
}

void interpolate_execute_split(const interpolate_plan plan, double *rin, double *iin, double *rout, double *iout)
{
  assert(SPLIT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  const int rounded_block_size = round16(block_size);
  double *const block_data = fftw_alloc_real(2 * 7 * rounded_block_size);
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

  fftw_free(block_data);
}

void interpolate_execute_split_product(const interpolate_plan plan, double *rin, double *iin, double *out)
{
  assert(SPLIT_PRODUCT == plan->interpolation);

  const int block_size = plan->dims[0] * plan->dims[1] * plan->dims[2];
  const int rounded_block_size = round16(block_size);
  double *const block_data = fftw_alloc_real(2 * 7 * rounded_block_size);
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

  fftw_free(block_data);
}

void interpolate_print_timings(const interpolate_plan plan)
{
  printf("Expand2: %f\n", time_point_delta(&plan->before_expand2, &plan->before_expand1));
  printf("Expand1: %f\n", time_point_delta(&plan->before_expand1, &plan->before_expand0));
  printf("Expand0: %f\n", time_point_delta(&plan->before_expand0, &plan->before_gather));
  printf("Gather: %f\n", time_point_delta(&plan->before_gather, &plan->end));
}

static void expand_dim0(interpolate_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch)
{
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1]; ++i1)
    {
      const size_t offset = i1*plan->strides[1] + i2*plan->strides[2];
      fftw_execute_dft(plan->dfts[0], in + offset, scratch);
      pointwise_multiply_complex(plan->dims[0], scratch, plan->rotations[0]);
      fftw_execute_dft(plan->idfts[0], scratch, out + offset);
    }
  }
}

static void expand_dim0_split(interpolate_plan plan, double *in[2], double *out[2], fftw_complex *scratch)
{
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1]; ++i1)
    {
      const size_t offset = i1*plan->strides[1] + i2*plan->strides[2];
      fftw_execute_split_dft(plan->dfts[0], in[0] + offset, in[1] + offset, (double*) scratch, ((double*) scratch) + 1);
      pointwise_multiply_complex(plan->dims[0], scratch, plan->rotations[0]);
      fftw_execute_split_dft(plan->idfts[0], ((double*) scratch) + 1, (double*) scratch, out[1] + offset, out[0] + offset);
    }
  }
}


static void expand_dim1(interpolate_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch)
{
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i0 + i2*plan->strides[2];
      fftw_execute_dft(plan->dfts[1], in + offset, scratch);
      pointwise_multiply_complex(plan->dims[1], scratch, plan->rotations[1]);
      fftw_execute_dft(plan->idfts[1], scratch, out + offset);
    }
  }
}

static void expand_dim1_split(interpolate_plan plan, double *in[2], double *out[2], fftw_complex *scratch)
{
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i0 + i2*plan->strides[2];
      fftw_execute_split_dft(plan->dfts[1], in[0] + offset, in[1] + offset, (double*) scratch, ((double*) scratch) + 1);
      pointwise_multiply_complex(plan->dims[1], scratch, plan->rotations[1]);
      fftw_execute_split_dft(plan->idfts[1], ((double*) scratch) + 1, (double*) scratch, out[1] + offset, out[0] + offset);

    }
  }
}

static void expand_dim2(interpolate_plan plan, fftw_complex *in, fftw_complex *out, fftw_complex *scratch)
{
  for(int i1=0; i1 < plan->dims[1]; ++i1)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i1*plan->strides[1] + i0;
      fftw_execute_dft(plan->dfts[2], in + offset, scratch);
      pointwise_multiply_complex(plan->dims[2], scratch, plan->rotations[2]);
      fftw_execute_dft(plan->idfts[2], scratch, out + offset);
    }
  }
}

static void expand_dim2_split(interpolate_plan plan, double *in[2], double *out[2], fftw_complex *scratch)
{
  for(int i1=0; i1 < plan->dims[1]; ++i1)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i1*plan->strides[1] + i0;
      fftw_execute_split_dft(plan->dfts[2], in[0] + offset, in[1] + offset, (double*) scratch, ((double*) scratch) + 1);
      pointwise_multiply_complex(plan->dims[2], scratch, plan->rotations[2]);
      fftw_execute_split_dft(plan->idfts[2], ((double*) scratch) + 1, (double*) scratch, out[1] + offset, out[0] + offset);
    }
  }
}
