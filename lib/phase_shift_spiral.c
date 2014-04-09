#include <tintl/interpolate.h>
#include <tintl/phase_shift_spiral.h>
#include <tintl/timer.h>
#include <tintl/allocation.h>
#include "common.h"
#include "fftw_utility.h"
#include <tintl/fftw_cycle.h>
#include "spiral_table.h"
#include <complex.h>
#include <stdint.h>
#include <fftw3.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

static const int TIMING_ITERATIONS = 50;
static const double pi = 3.14159265358979323846;

typedef enum
{
  PACKED,
  SEPARATE
} packing_strategy_t;

/// Implementation-specific structure for phase-shift interpolation plans.
typedef struct
{
  struct interpolate_plan_s common;
  packing_strategy_t packing_strategy;
  spiral_interpolate_function_t packed_interpolate;
} phase_shift_plan_s;


typedef phase_shift_plan_s *phase_shift_plan;

static interpolate_plan allocate_plan(void);

/* Interface functions */
static const char *get_name(interpolate_plan plan);
static void phase_shift_set_flags(interpolate_plan plan, const int flags);
static void phase_shift_get_statistic_float(const interpolate_plan plan, int statistic, int index, stat_type_t *type, double *value);
static void phase_shift_interpolate_execute_interleaved(interpolate_plan plan, fftw_complex *in, fftw_complex *out);
static void phase_shift_interpolate_execute_split(interpolate_plan plan, double *rin, double *iin, double *rout, double *iout);
static void phase_shift_interpolate_execute_split_product(interpolate_plan plan, double *rin, double *iin, double *out);
static void phase_shift_interpolate_print_timings(interpolate_plan plan);
static void phase_shift_interpolate_destroy_detail(interpolate_plan plan);

static phase_shift_plan plan_common(interpolation_t type, int n0, int n1, int n2, int flags);

static const char *get_name(interpolate_plan plan)
{
  return "phase_shift-spiral";
}

static void phase_shift_set_flags(interpolate_plan parent, const int flags)
{
}

static void phase_shift_get_statistic_float(const interpolate_plan parent, const int statistic, const int index, stat_type_t *type, double *value)
{
  *type = STATISTIC_UNKNOWN;
}

static interpolate_plan allocate_plan(void)
{
  setup_threading();

  interpolate_plan holder = malloc(sizeof(phase_shift_plan_s));
  if (holder == NULL)
    return NULL;

  holder->get_name = get_name;
  holder->set_flags = phase_shift_set_flags;
  holder->get_statistic_float = phase_shift_get_statistic_float;
  holder->execute_interleaved = phase_shift_interpolate_execute_interleaved;
  holder->execute_split = phase_shift_interpolate_execute_split;
  holder->execute_split_product = phase_shift_interpolate_execute_split_product;
  holder->print_timings = phase_shift_interpolate_print_timings;
  holder->destroy_detail = phase_shift_interpolate_destroy_detail;

  return holder;
}

static phase_shift_plan plan_common(interpolation_t type, int n0, int n1, int n2, int flags)
{
  spiral_interpolate_function_t interpolate_function = 0;
  for(size_t index = 0; index < spiral_function_table_size; ++index)
  {
    spiral_function_info_t *const entry = &spiral_function_table[index];
    if (entry->x == n0 && entry->y == n1 && entry->z == n2)
    {
      entry->initialise();
      interpolate_function = entry->interpolate;
    }
  }

  if (interpolate_function != NULL)
  {
    interpolate_plan parent = allocate_plan();
    if (parent == NULL)
      return NULL;

    phase_shift_plan plan = (phase_shift_plan) parent;
    populate_properties(parent, type, n0, n1, n2);
    plan->packed_interpolate = interpolate_function;
    plan->packing_strategy = PACKED;
    return plan;
  }
  else
  {
    return NULL;
  }
}

interpolate_plan interpolate_plan_3d_phase_shift_spiral_interleaved(int n0, int n1, int n2, int flags)
{
  phase_shift_plan plan = plan_common(INTERPOLATE_INTERLEAVED, n0, n1, n2, flags);
  return cast_to_parent(plan);
}

interpolate_plan interpolate_plan_3d_phase_shift_spiral_split(int n0, int n1, int n2, int flags)
{
  phase_shift_plan plan = plan_common(INTERPOLATE_SPLIT, n0, n1, n2, flags);
  return cast_to_parent(plan);
}

interpolate_plan interpolate_plan_3d_phase_shift_spiral_product(int n0, int n1, int n2, int flags)
{
  interpolate_plan parent = interpolate_plan_3d_phase_shift_spiral_split(n0, n1, n2, flags);
  if (parent == NULL)
    return NULL;

  parent->type = INTERPOLATE_SPLIT_PRODUCT;
  return parent;
}

static void phase_shift_interpolate_destroy_detail(interpolate_plan plan)
{
}

static void phase_shift_interpolate_execute_interleaved(interpolate_plan parent, fftw_complex *in, fftw_complex *out)
{
  assert(INTERPOLATE_INTERLEAVED == parent->type);

  phase_shift_plan plan = (phase_shift_plan) parent;
  assert(plan->packing_strategy == PACKED);

  plan->packed_interpolate((double*) out, (double*) in);
}

static void phase_shift_interpolate_execute_split(interpolate_plan parent, double *rin, double *iin, double *rout, double *iout)
{
  assert(INTERPOLATE_SPLIT == parent->type);

  phase_shift_plan plan = (phase_shift_plan) parent;
  block_info_t coarse_info, fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);
  const size_t block_size = num_elements_block(&coarse_info);

  fftw_complex *const scratch_coarse = tinl_alloc_complex(block_size);
  fftw_complex *const scratch_fine = tinl_alloc_complex(8 * block_size);

  interleave_real(block_size, (double*) scratch_coarse, rin, iin);
  plan->packed_interpolate((double*) scratch_fine, (double*) scratch_coarse);
  deinterleave_real(8 * block_size, (const double*) scratch_fine, rout, iout);

  tinl_free(scratch_coarse);
  tinl_free(scratch_fine);
}

void phase_shift_interpolate_execute_split_product(interpolate_plan parent, double *rin, double *iin, double *out)
{
  assert(INTERPOLATE_SPLIT_PRODUCT == parent->type);

  phase_shift_plan plan = (phase_shift_plan) parent;
  block_info_t coarse_info, fine_info;
  get_block_info_coarse(parent, &coarse_info);
  get_block_info_fine(parent, &fine_info);
  const size_t block_size = num_elements_block(&coarse_info);

  fftw_complex *const scratch_coarse = tinl_alloc_complex(block_size);
  fftw_complex *const scratch_fine = tinl_alloc_complex(8 * block_size);

  interleave_real(block_size, (double*) scratch_coarse, rin, iin);
  plan->packed_interpolate((double*) scratch_fine, (double*) scratch_coarse);
  complex_to_product(8 * block_size, scratch_fine, out);

  tinl_free(scratch_coarse);
  tinl_free(scratch_fine);
}

void phase_shift_interpolate_print_timings(interpolate_plan plan)
{
}
