#ifndef PADDING_AWARE_H
#define PADDING_AWARE_H

#include "interpolate.h"

interpolate_plan interpolate_plan_3d_padding_aware_interleaved(int n0, int n1, int n2, int flags);
interpolate_plan interpolate_plan_3d_padding_aware_split(int n0, int n1, int n2, int flags);
interpolate_plan interpolate_plan_3d_padding_aware_product(int n0, int n1, int n2, int flags);

#endif
