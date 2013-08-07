#ifndef PLAN_CACHE_H
#define PLAN_CACHE_H

#include "common.h"
#include <interpolate.h>

struct plan_cache_entry_s;

/// Key for plan cache.
typedef struct
{
  int n0;
  int n1;
  int n2;
  interpolation_t type;
} plan_key_t;

/// Plan cache structure.
typedef struct
{
  size_t size;
  struct plan_cache_entry_s *first;

} plan_cache_t;

/// Initialise a plan cache.
void plan_cache_init(plan_cache_t *cache);

/// Insert an entry into the plan cache.
int plan_cache_insert(plan_cache_t *cache, const plan_key_t *key, interpolate_plan plan);

/// Retrieve an entry from a plan cache, returning NULL if absent.
interpolate_plan plan_cache_get(plan_cache_t *cache, const plan_key_t *key);

#endif
