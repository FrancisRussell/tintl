#ifndef PLAN_CACHE_H
#define PLAN_CACHE_H

#include "common.h"
#include <interpolate.h>

struct plan_cache_entry_s;

typedef struct
{
  int n0;
  int n1;
  int n2;
  interpolation_t type;
} plan_key_t;

typedef struct
{
  size_t size;
  struct plan_cache_entry_s *first;

} plan_cache_t;

void plan_cache_init(plan_cache_t *cache);
int plan_cache_insert(plan_cache_t *cache, const plan_key_t *key, interpolate_plan plan);
interpolate_plan plan_cache_get(plan_cache_t *cache, const plan_key_t *key);

#endif
