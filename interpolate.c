#include "interpolate.h"
#include <complex.h>
#include <fftw3.h>
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifdef __SSE2__
#include <emmintrin.h>
#endif

static void expand_dim0(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void expand_dim1(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void expand_dim2(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void build_rotation(int size, fftw_complex *out);
static void gather_blocks(interpolate_plan plan, fftw_complex *blocks[8], fftw_complex *out);
static void pointwise_multiply_complex(int size, fftw_complex *a, fftw_complex *b);
static void interleave_complex(int size, fftw_complex *out, const fftw_complex *even, const fftw_complex *odd);

interpolate_plan plan_interpolate_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out)
{
  int flags = 0;
  interpolate_plan_s *const plan = malloc(sizeof(interpolate_plan_s));

  plan->dims[0] = n0;
  plan->dims[1] = n1;
  plan->dims[2] = n2;

  plan->strides[0] = 1;
  plan->strides[1] = n0;
  plan->strides[2] = n0 * n1;

  for(int dim = 0; dim < 3; ++dim)
  {
    plan->rotations[dim] = fftw_alloc_complex(plan->dims[dim]);
    build_rotation(plan->dims[dim], plan->rotations[dim]);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    plan->dfts[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1, 
      in, NULL, plan->strides[dim], 0, 
      out, NULL, plan->strides[dim], 0, 
      FFTW_FORWARD, flags);
  }

  for(int dim=0; dim < 3; ++dim)
  {
    plan->idfts[dim] = fftw_plan_many_dft(1, &plan->dims[dim], 1, 
      out, NULL, plan->strides[dim], 0, 
      out, NULL, plan->strides[dim], 0, 
      FFTW_BACKWARD, flags);
  }

  return plan;
}

void interpolate_destroy_plan(interpolate_plan plan)
{
  for(int dim = 0; dim < 3; ++dim)
    fftw_free(plan->rotations[dim]);

  free(plan);
}

static void build_rotation(int size, fftw_complex *out)
{
  const double pi = 3.14159265358979323846;
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
  for(int i = 0; i < size; ++i)
  {
    out[i*2] = even[i];
    out[i*2 + 1] = odd[i];
  }
}

static void pointwise_multiply_complex(int size, fftw_complex *a, fftw_complex *b)
{
#if __SSE2__
  const __m128d neg = _mm_setr_pd(-1.0, 1.0);
  for(int i=0; i<size; ++i)
  {
    __m128d a_vec, a_imag, a_real, b_vec, res;
    a_vec = _mm_load_pd((double*)(a + i));
    b_vec = _mm_load_pd((double*)(b + i));
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

static void gather_blocks(interpolate_plan plan, fftw_complex *blocks[8], fftw_complex *out)
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

void interpolate_execute(const interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  fftw_complex *blocks[8];
  blocks[0] = in;

  for(int block = 1; block < 8; ++block)
    blocks[block] = fftw_alloc_complex(plan->dims[0] * plan->dims[1] * plan->dims[2]);

  expand_dim2(plan, blocks[0], blocks[1]);

  for(int n = 0; n < 2; ++n)
    expand_dim1(plan, blocks[n], blocks[n + 2]);

  for(int n = 0; n < 4; ++n)
    expand_dim0(plan, blocks[n], blocks[n + 4]);

  gather_blocks(plan, blocks, out);

  for(int block = 1; block < 8; ++block)
    fftw_free(blocks[block]);
}

static void expand_dim0(interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1]; ++i1)
    {
      const size_t offset = i1*plan->strides[1] + i2*plan->strides[2];
      fftw_execute_dft(plan->dfts[0], in + offset, out + offset);
      pointwise_multiply_complex(plan->dims[0], out + offset, plan->rotations[0]);
      fftw_execute_dft(plan->idfts[0], out + offset, out + offset);
    }
  }
}

static void expand_dim1(interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  for(int i2=0; i2 < plan->dims[2]; ++i2)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i0 + i2*plan->strides[2];
      fftw_execute_dft(plan->dfts[1], in + offset, out + offset);

      for(int i1=0; i1 < plan->dims[1]; ++i1)
        (out + offset)[i1*plan->strides[1]] *= plan->rotations[1][i1];

      fftw_execute_dft(plan->idfts[1], out + offset, out + offset);
    }
  }
}

static void expand_dim2(interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  for(int i1=0; i1 < plan->dims[1]; ++i1)
  {
    for(int i0=0; i0 < plan->dims[0]; ++i0)
    {
      const size_t offset = i1*plan->strides[1] + i0;
      fftw_execute_dft(plan->dfts[2], in + offset, out + offset);

      for(int i2 = 0; i2 < plan->dims[2]; ++i2)
        (out + offset)[i2*plan->strides[2]] *= plan->rotations[2][i2];

      fftw_execute_dft(plan->idfts[2], out + offset, out + offset);
    }
  }
}

static double test_function(double x, double y, double z)
{
  return sin(2 * x) + cos(3 * y) + sin(4 * z);
}

int main(int argc, char **argv)
{
  const double pi = 3.14159265358979323846;
  const int width = 100;
  fftw_complex *in = fftw_alloc_complex(width * width * width);
  fftw_complex *out = fftw_alloc_complex(8 * width * width * width);
  interpolate_plan plan = plan_interpolate_3d(width, width, width, in, out);


  for(int x=0; x < width; ++x)
  {
    for(int y=0; y < width; ++y)
    {
      for(int z=0; z < width; ++z)
      {
        const int offset = x*width*width + y*width + z;
        const double x_pos = (x * 2.0 * pi)/width;
        const double y_pos = (y * 2.0 * pi)/width;
        const double z_pos = (z * 2.0 * pi)/width;

        in[offset] = test_function(x_pos, y_pos, z_pos);

        //printf("in[%d][%d][%d] = %f\n", x, y, z, in[offset][0]);
      }
    }
  }

  interpolate_execute(plan, in, out);

  double abs_val = 0.0;

  for(int x=0; x < 2 * width; ++x)
  {
    for(int y=0; y < 2 * width; ++y)
    {
      for(int z=0; z < 2 * width; ++z)
      {
        const int offset = 4*x*width*width + 2*y*width + z;
        const double x_pos = (x * pi)/width;
        const double y_pos = (y * pi)/width;
        const double z_pos = (z * pi)/width;

        const double expected = test_function(x_pos, y_pos, z_pos);
        //printf("out(e)[%d][%d][%d] = %f\n", x, y, z, expected);
        //printf("out(a)[%d][%d][%d] = %f\n", x, y, z, out[offset][0]);

        abs_val += cabs(out[offset] - expected);
      }
    }
  }

  printf("Delta: %f\n", abs_val);

  interpolate_destroy_plan(plan);

  return EXIT_SUCCESS;
}
