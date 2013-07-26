#include <complex.h>
#include <float.h>
#include <assert.h>
#include <stdlib.h>
#include <fftw3.h>
#include <interpolate.h>
#include "common.h"
#include <naive.h>
#include <padding_aware.h>
#include <phase_shift.h>

const char *interpolate_get_name(const interpolate_plan plan)
{
  return plan->get_name(plan->detail);
}

void interpolate_execute_interleaved(const interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  plan->execute_interleaved(plan->detail, in, out);
}

void interpolate_execute_split(const interpolate_plan plan, double *rin, double *iin, double *rout, double *iout)
{
  plan->execute_split(plan->detail, rin, iin, rout, iout);
}

void interpolate_execute_split_product(const interpolate_plan plan, double *rin, double *iin, double *out)
{
  plan->execute_split_product(plan->detail, rin, iin, out);
}

void interpolate_print_timings(const interpolate_plan plan)
{
  plan->print_timings(plan->detail);
}

void interpolate_destroy_plan(interpolate_plan plan)
{
  plan->destroy_detail(plan->detail);
  free(plan);
}

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

static interpolate_plan find_best_plan(double (*timer)(interpolate_plan, const int*),
  plan_constructor_t *constructors,
  int n0, int n1, int n2, int flags)
{
  const int dims[] = {n0, n1, n2};
  int plan_type_count = 0;

  while(constructors[plan_type_count] != NULL)
    ++plan_type_count;

  interpolate_plan best_plan = NULL;
  double best_time = DBL_MAX;

  for(size_t plan_id = 0; plan_id < plan_type_count; ++plan_id)
  {
    interpolate_plan plan = constructors[plan_id](n0, n1, n2, flags);
    const double time = timer(plan, dims);

    if (time < best_time)
    {
      best_time = time;
      if (best_plan != NULL)
        interpolate_destroy_plan(best_plan);
      best_plan = plan;
    }
    else
    {
      interpolate_destroy_plan(plan);
    }
  }

  assert(best_plan != NULL);
  return best_plan;
}

interpolate_plan interpolate_plan_3d_interleaved_best(int n0, int n1, int n2, int flags)
{
  plan_constructor_t constructors[] = {
    interpolate_plan_3d_naive_interleaved,
    interpolate_plan_3d_padding_aware_interleaved,
    interpolate_plan_3d_phase_shift_interleaved,
    NULL
  };

  return find_best_plan(time_interpolate_interleaved, constructors, n0, n1, n2, flags);
}

interpolate_plan interpolate_plan_3d_split_best(int n0, int n1, int n2, int flags)
{
  plan_constructor_t constructors[] = {
    interpolate_plan_3d_naive_split,
    interpolate_plan_3d_padding_aware_split,
    interpolate_plan_3d_phase_shift_split,
    NULL
  };
  return find_best_plan(time_interpolate_split, constructors, n0, n1, n2, flags);
}

interpolate_plan interpolate_plan_3d_split_product_best(int n0, int n1, int n2, int flags)
{
  plan_constructor_t constructors[] = {
    interpolate_plan_3d_naive_product,
    interpolate_plan_3d_padding_aware_product,
    interpolate_plan_3d_phase_shift_product,
    NULL
  };
  return find_best_plan(time_interpolate_split_product, constructors, n0, n1, n2, flags);
}
