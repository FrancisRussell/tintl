#ifndef FFTW_UTILITY_H
#define FFTW_UTILITY_H

#include <fftw3.h>

static inline void fftw_destroy_plan_maybe_null(fftw_plan plan)
{
  if (plan != NULL)
    fftw_destroy_plan(plan);
}

#endif
