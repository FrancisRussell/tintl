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
#include "plan_cache.h"

static const int best_plan_cache_enabled = 1;

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
  --plan->ref_cnt;

  if (plan->ref_cnt == 0)
  {
    plan->destroy_detail(plan->detail);
    free(plan);
  }
}

static plan_cache_t *get_best_plan_cache(void)
{
  static int cache_initialised = 0;
  static plan_cache_t cache;

  if (cache_initialised == 0)
  {
    plan_cache_init(&cache);
    cache_initialised = 1;
  }

  return &cache;
}

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

static interpolate_plan find_best_plan(double (*timer)(interpolate_plan, const int*),
  plan_constructor_t *constructors,
  int n0, int n1, int n2, interpolation_t type, int flags)
{
  plan_cache_t *best_plan_cache = get_best_plan_cache();
  assert(best_plan_cache != NULL);

  plan_key_t key;
  key.n0 = n0;
  key.n1 = n1;
  key.n2 = n2;
  key.type = type;

  if (best_plan_cache_enabled)
  {
    interpolate_plan cached_plan = plan_cache_get(best_plan_cache, &key);
    if (cached_plan != NULL)
    {
      ++cached_plan->ref_cnt;
      return cached_plan;
    }
  }

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
  if (best_plan_cache_enabled)
  {
    const int inserted = plan_cache_insert(best_plan_cache, &key, best_plan);
    if (inserted)
      ++best_plan->ref_cnt;
  }
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

  return find_best_plan(time_interpolate_interleaved, constructors, n0, n1, n2, INTERLEAVED, flags);
}

interpolate_plan interpolate_plan_3d_split_best(int n0, int n1, int n2, int flags)
{
  plan_constructor_t constructors[] = {
    interpolate_plan_3d_naive_split,
    interpolate_plan_3d_padding_aware_split,
    interpolate_plan_3d_phase_shift_split,
    NULL
  };
  return find_best_plan(time_interpolate_split, constructors, n0, n1, n2, SPLIT, flags);
}

interpolate_plan interpolate_plan_3d_split_product_best(int n0, int n1, int n2, int flags)
{
  plan_constructor_t constructors[] = {
    interpolate_plan_3d_naive_product,
    interpolate_plan_3d_padding_aware_product,
    interpolate_plan_3d_phase_shift_product,
    NULL
  };
  return find_best_plan(time_interpolate_split_product, constructors, n0, n1, n2, SPLIT_PRODUCT, flags);
}
