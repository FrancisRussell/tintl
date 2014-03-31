#ifndef TINTL_NAIVE_H
#define TINTL_NAIVE_H

/// \file
/// Functions for construction of interpolation plans using the naïve
/// technique.

#include "interpolate.h"

/// Construct interleaved interpolation plan.
interpolate_plan interpolate_plan_3d_naive_interleaved(int n0, int n1, int n2, int flags);

/// Construct split interpolation plan.
interpolate_plan interpolate_plan_3d_naive_split(int n0, int n1, int n2, int flags);

/// Construct split-product interpolation plan.
interpolate_plan interpolate_plan_3d_naive_product(int n0, int n1, int n2, int flags);

#endif
