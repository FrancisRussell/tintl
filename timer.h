#ifndef TIMER_H
#define TIMER_H

#include <sys/time.h>

typedef struct
{
  struct timeval val;
} time_point_t;

void time_point_save(time_point_t *point);
double time_point_delta(const time_point_t *t1, const time_point_t *t2);

#endif
