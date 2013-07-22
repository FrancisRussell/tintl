#include "timer.h"
#include "allocation.h"
#include "interpolate.h"
#include "phase_shift.h"
#include "naive.h"
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

static const double pi = 3.14159265358979323846;

static double test_function(double x, double y, double z)
{
  return sin(2 * x) + cos(3 * y) + sin(4 * z);
}

static void perform_timing(plan_constructor_t constructor,
  const int x_width, const int y_width, const int z_width)
{
  time_point_t begin_resample, end_resample, begin_plan, end_plan;

  fftw_complex *in = rs_alloc_complex(x_width * y_width * z_width);
  fftw_complex *out = rs_alloc_complex(8 * x_width * y_width * z_width);

  time_point_save(&begin_plan);
  interpolate_plan plan = constructor(x_width, y_width, z_width, 0);
  time_point_save(&end_plan);


  for(int x=0; x < x_width; ++x)
  {
    for(int y=0; y < y_width; ++y)
    {
      for(int z=0; z < z_width; ++z)
      {
        const int offset = x*y_width*z_width + y*z_width + z;
        const double x_pos = (x * 2.0 * pi)/x_width;
        const double y_pos = (y * 2.0 * pi)/y_width;
        const double z_pos = (z * 2.0 * pi)/z_width;

        in[offset] = test_function(x_pos, y_pos, z_pos);
      }
    }
  }

  time_point_save(&begin_resample);
  interpolate_execute_interleaved(plan, in, out);
  time_point_save(&end_resample);

  double abs_val = 0.0;

  for(int x=0; x < 2 * x_width; ++x)
  {
    for(int y=0; y < 2 * y_width; ++y)
    {
      for(int z=0; z < 2 * z_width; ++z)
      {
        const int offset = 4*x*y_width*z_width + 2*y*z_width + z;
        const double x_pos = (x * pi)/x_width;
        const double y_pos = (y * pi)/y_width;
        const double z_pos = (z * pi)/z_width;

        const double expected = test_function(x_pos, y_pos, z_pos);
        abs_val += cabs(out[offset] - expected);
      }
    }
  }

  printf("Interpolation variant: %s\n", interpolate_get_name(plan));
  printf("Problem size: %d x %d x %d\n", x_width, y_width, z_width);
  interpolate_print_timings(plan);
  printf("Planning time: %f\n", time_point_delta(&begin_plan, &end_plan));
  printf("Execution time: %f\n", time_point_delta(&begin_resample, &end_resample));
  printf("Delta: %f\n", abs_val);

  rs_free(in);
  rs_free(out);
  interpolate_destroy_plan(plan);
}

int main(int argc, char **argv)
{
  perform_timing(interpolate_plan_3d_naive_interleaved, 75, 75, 75);
  printf("\n");
  perform_timing(interpolate_plan_3d_phase_shift_interleaved, 75, 75, 75);
  return EXIT_SUCCESS;
}
