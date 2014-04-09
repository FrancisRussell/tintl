#ifndef TINTL_PHASE_SHIFT_SPIRAL_H
#define TINTLPHASE_SHIFT_SPIRAL_H

/// \file
/// Functions for construction of interpolation plans using the
/// phase-shift technique with SPIRAL-generated code.

#include "interpolate.h"

/// Construct interleaved interpolation plan.
interpolate_plan interpolate_plan_3d_phase_shift_spiral_interleaved(int n0, int n1, int n2, int flags);

/// Construct split interpolation  plan.
interpolate_plan interpolate_plan_3d_phase_shift_spiral_split(int n0, int n1, int n2, int flags);

/// Construct split-product interpolation  plan.
interpolate_plan interpolate_plan_3d_phase_shift_spiral_product(int n0, int n1, int n2, int flags);

#endif
