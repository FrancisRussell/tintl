#ifndef PADDING_AWARE_H
#define PADDING_AWARE_H

/// \file
/// Functions for construction of interpolation plans using the
/// padding-aware technique.

#include "interpolate.h"


/// Construct interleaved interpolation plan.
interpolate_plan interpolate_plan_3d_padding_aware_interleaved(int n0, int n1, int n2, int flags);

/// Construct split interpolation plan.
interpolate_plan interpolate_plan_3d_padding_aware_split(int n0, int n1, int n2, int flags);

/// Construct split-product interpolation plan.
interpolate_plan interpolate_plan_3d_padding_aware_product(int n0, int n1, int n2, int flags);

#endif
