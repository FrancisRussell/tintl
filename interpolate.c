#include "interpolate.h"
#include <fftw3.h>
#include <cblas.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static void expand_dim0(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void expand_dim1(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void expand_dim2(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void build_rotation(int size, fftw_complex *out);
static void gather_blocks(interpolate_plan plan, fftw_complex *blocks[8], fftw_complex *out);

static inline void mul_complex(fftw_complex *out, fftw_complex *a, fftw_complex *b)
{
  const double r = (*a)[0]*(*b)[0] - (*a)[1]*(*b)[1];
  const double i = (*a)[0]*(*b)[1] + (*a)[1]*(*b)[0];
  (*out)[0] = r;
  (*out)[1] = i;
}

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
      out[freq][0] = out[freq][1] = 0.0;
    }
    else
    {
      double theta;
      if (freq < 1 + size / 2)
        theta = theta_base * freq;
      else
        theta = pi + (theta_base * freq);

      out[freq][0] = cos(theta) / size;
      out[freq][1] = sin(theta) / size; 
    }
  }
}

static void gather_blocks(interpolate_plan plan, fftw_complex *blocks[8], fftw_complex *out)
{
  for(int i2=0; i2 < plan->dims[2] * 2; ++i2)
  {
    for(int i1=0; i1 < plan->dims[1] * 2; ++i1)
    {
      for(int i0=0; i0 < plan->dims[0] * 2; ++i0)
      {
        fftw_complex *block = blocks[(i2 % 2) + (i1 % 2) * 2 + (i0 % 2) * 4];
        const int in_offset = (i2/2) * plan->strides[2] + (i1/2) * plan->strides[1] + (i0 / 2);
        const int out_offset = i2 * plan->strides[2] * 4 + i1 * plan->strides[1] * 2 + i0;
        out[out_offset][0] = block[in_offset][0];
        out[out_offset][1] = block[in_offset][1];
      }
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

  expand_dim1(plan, blocks[0], blocks[2]);
  expand_dim1(plan, blocks[1], blocks[3]);

  expand_dim0(plan, blocks[0], blocks[4]);
  expand_dim0(plan, blocks[1], blocks[5]);
  expand_dim0(plan, blocks[2], blocks[6]);
  expand_dim0(plan, blocks[3], blocks[7]);

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

      for(int i0=0; i0 < plan->dims[0]; ++i0)
        mul_complex(out + offset + i0, out + offset + i0, &(plan->rotations[0][i0]));

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
        mul_complex(out + offset + i1*plan->strides[1], out + offset + i1*plan->strides[1], &(plan->rotations[1][i1]));

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
        mul_complex(out + offset + i2*plan->strides[2], out + offset + i2*plan->strides[2], &(plan->rotations[2][i2]));

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
  const int width = 10;
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

        in[offset][0] = test_function(x_pos, y_pos, z_pos);
        in[offset][1] = 0.0;

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

        abs_val += abs(out[offset][0] - expected);
      }
    }
  }

  printf("Delta: %f\n", abs_val);

  interpolate_destroy_plan(plan);

  return EXIT_SUCCESS;
}
