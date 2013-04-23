#include "timer.h"
#include <assert.h>
#include <sys/time.h>
#include <stdlib.h>

void time_point_save(time_point_t *point)
{
  const int result = gettimeofday(&point->val, NULL);
  assert(result == 0);
}

double time_point_delta(const time_point_t *t1, const time_point_t *t2)
{
  const double secs = t2->val.tv_sec - t1->val.tv_sec;
  const double usecs = (t2->val.tv_usec - t1->val.tv_usec)/1000000.0;
  return secs + usecs;
}
