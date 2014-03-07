#include "timer.h"
#include <assert.h>
#include <time.h>
#include <stdlib.h>

void time_point_save(time_point_t *point)
{
  const int result = clock_gettime(CLOCK_MONOTONIC, &point->val);
  assert(result == 0);
}

double time_point_delta(const time_point_t *t1, const time_point_t *t2)
{
  const double secs = t2->val.tv_sec - t1->val.tv_sec;
  const double nsecs = (t2->val.tv_nsec - t1->val.tv_nsec)/1000000000.0;
  return secs + nsecs;
}
