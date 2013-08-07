#ifndef TIMER_H
#define TIMER_H

/// \file
/// Functions for performing wall-clock timings in seconds.

#include <sys/time.h>

/// Opaque time representation.
typedef struct
{
  struct timeval val;
} time_point_t;

/// Record the current time.
void time_point_save(time_point_t *point);

/// Calculate the delta t2-t1 in seconds.
double time_point_delta(const time_point_t *t1, const time_point_t *t2);

#endif
