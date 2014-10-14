#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "tintl/naive.h"
#include "tintl/padding_aware.h"
#include "tintl/phase_shift.h"
#include "tintl/phase_shift_spiral.h"
#include "storage.h"
#include "papi_multiplex.h"

typedef interpolate_plan (*plan_constructor_t)(int n0, int n1, int n2, int flags);

void do_profiling(const int *sizes, const size_t count)
{
  const int repetitions = 50;
  const int size_field_width = 5;
  const int result_field_width = 15;

  const plan_constructor_t plan_constructors[] =
  {
    interpolate_plan_3d_phase_shift_split,
    interpolate_plan_3d_phase_shift_spiral_split,
  };

  const papi_event_t counters[] =
  {
    PAPI_TOT_CYC,
    PAPI_REF_CYC,
    PAPI_TOT_INS,
    PAPI_STL_ICY,
    PAPI_L2_TCM,
    PAPI_L3_TCM,
    PAPI_DP_OPS
  };

  const size_t num_plan_constructors = sizeof(plan_constructors)/sizeof(plan_constructors[0]);
  const size_t num_counters = sizeof(counters)/sizeof(counters[0]);

  printf("%*s", size_field_width, "");
  for(size_t constructor_index = 0; constructor_index < num_plan_constructors; ++constructor_index)
  {
    interpolate_plan plan = plan_constructors[constructor_index](75, 75, 75, 0);
    assert(plan != NULL);
    printf("%*s", (int) (result_field_width * num_counters), interpolate_get_name(plan));
    interpolate_destroy_plan(plan);
  }
  printf("\n");

  printf("%*s", size_field_width, "");
  for(size_t constructor_index = 0; constructor_index < num_plan_constructors; ++constructor_index)
  {
    char name[PAPI_MAX_STR_LEN];
    for(size_t i=0; i < num_counters; ++i)
    {
      PAPI_CHECK(PAPI_event_code_to_name(counters[i], name));
      printf("%*s", result_field_width, name);
    }
  }
  printf("\n");

  for(size_t size = 7; size <= 119; size += 2)
  {
    const int layout = SPLIT;
    const size_t block_size = size * size * size;
    storage_t coarse, fine;

    storage_allocate(&coarse, layout, block_size);
    storage_allocate(&fine, layout, 8 * block_size);

    printf("%5zd", size);

    for(size_t constructor_index = 0; constructor_index < num_plan_constructors; ++constructor_index)
    {
      interpolate_plan plan = plan_constructors[constructor_index](size, size, size, 0);
      assert(plan != NULL);

      papi_multiplex_t multiplex;
      PAPI_CHECK(pm_init(&multiplex));

      for(size_t i=0; i < num_counters; ++i)
        PAPI_CHECK(pm_add_event(&multiplex, counters[i]));

      for(int repetition = 0; repetition < repetitions; ++repetition)
      {
        PAPI_CHECK(pm_start(&multiplex));
        execute_interpolate(plan, &coarse, &fine);
        PAPI_CHECK(pm_stop(&multiplex));
      }

      for(size_t i=0; i < num_counters; ++i)
      {
        long long value;
        PAPI_CHECK(pm_count(&multiplex, counters[i], &value));
        printf("%15lld", value);
      }

      PAPI_CHECK(pm_destroy(&multiplex));
      interpolate_destroy_plan(plan);
    }

    printf("\n");

    storage_free(&coarse);
    storage_free(&fine);
  }
}

int main(void)
{
  const int sizes[] = {70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130};
  do_profiling(sizes, sizeof(sizes)/sizeof(int));
  return EXIT_SUCCESS;
}
