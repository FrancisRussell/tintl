#ifndef PHASE_SHIFT_H
#define PHASE_SHIFT_H

#include "interpolate.h"

interpolate_plan interpolate_plan_3d_phase_shift_interleaved(int n0, int n1, int n2, int flags);
interpolate_plan interpolate_plan_3d_phase_shift_split(int n0, int n1, int n2, int flags);
interpolate_plan interpolate_plan_3d_phase_shift_product(int n0, int n1, int n2, int flags);

#endif
