#ifndef PHASE_SHIFT_H
#define PHASE_SHIFT_H

/// \file
/// Functions for construction of interpolation plans using the
/// phase-shift technique.

#include "interpolate.h"

/// Construct interleaved interpolation plan.
interpolate_plan interpolate_plan_3d_phase_shift_interleaved(int n0, int n1, int n2, int flags);

/// Construct split interpolation  plan.
interpolate_plan interpolate_plan_3d_phase_shift_split(int n0, int n1, int n2, int flags);

/// Construct split-product interpolation  plan.
interpolate_plan interpolate_plan_3d_phase_shift_product(int n0, int n1, int n2, int flags);

#endif
