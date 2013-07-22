#ifndef NAIVE_H
#define NAIVE_H

#include "interpolate.h"

interpolate_plan interpolate_plan_3d_naive_interleaved(int n0, int n1, int n2, int flags);
interpolate_plan interpolate_plan_3d_naive_split(int n0, int n1, int n2, int flags);
interpolate_plan interpolate_plan_3d_naive_product(int n0, int n1, int n2, int flags);

#endif
