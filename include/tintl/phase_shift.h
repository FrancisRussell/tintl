#ifndef TINTL_PHASE_SHIFT_H
#define TINTL_PHASE_SHIFT_H

/// \file
/// Functions for construction of interpolation plans using the
/// phase-shift technique.

#include "interpolate.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// Construct interleaved interpolation plan.
interpolate_plan interpolate_plan_3d_phase_shift_interleaved(int n0, int n1, int n2, int flags);

/// Construct split interpolation  plan.
interpolate_plan interpolate_plan_3d_phase_shift_split(int n0, int n1, int n2, int flags);

/// Construct split-product interpolation  plan.
interpolate_plan interpolate_plan_3d_phase_shift_product(int n0, int n1, int n2, int flags);

enum
{
  PHASE_SHIFT_STATISTIC_BATCH_TRANSFORMS = STATISTIC_LAST_COMMON_VALUE,
  PHASE_SHIFT_STATISTIC_INDIVIDUAL_TRANSFORMS
};

#ifdef __cplusplus
}
#endif

#endif
