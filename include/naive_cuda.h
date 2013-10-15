#ifndef NAIVE_CUDA_H
#define NAIVE_CUDA_H

/// \file
/// Functions for construction of interpolation plans using the naïve
/// technique using CUDA.

#include "interpolate.h"

#ifdef __cplusplus
extern "C"
{
#endif

/// Construct interleaved interpolation plan.
interpolate_plan interpolate_plan_3d_naive_cuda_interleaved(int n0, int n1, int n2, int flags);

/// Construct split interpolation plan.
interpolate_plan interpolate_plan_3d_naive_cuda_split(int n0, int n1, int n2, int flags);

/// Construct split-product interpolation plan.
interpolate_plan interpolate_plan_3d_naive_cuda_product(int n0, int n1, int n2, int flags);

#ifdef __cplusplus
}
#endif

#endif
