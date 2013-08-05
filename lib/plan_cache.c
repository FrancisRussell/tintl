#include "plan_cache.h"
#include <string.h>
#include <stdlib.h>

typedef struct plan_cache_entry_s
{
  plan_key_t key;
  interpolate_plan value;
  struct plan_cache_entry_s *next;
} plan_cache_entry_t;

void plan_cache_init(plan_cache_t *cache)
{
  cache->size = 0;
  cache->first = NULL;
}

static void copy_key(const plan_key_t *from, plan_key_t *to)
{
  memcpy(to, from, sizeof(plan_key_t));
}

static int compare_int(const int a, const int b)
{
  if (a < b)
    return -1;
  else if (a > b)
    return 1;
  else
    return 0;
}

static int plan_compare_key(const plan_key_t *first, const plan_key_t *second)
{
  if (first->n0 != second->n0) return compare_int(first->n0, second->n0);
  if (first->n1 != second->n1) return compare_int(first->n1, second->n1);
  if (first->n2 != second->n2) return compare_int(first->n2, second->n2);
  if (first->type != second->type) return compare_int(first->type, second->type);
  return 0;
}

int plan_cache_insert(plan_cache_t *cache, const plan_key_t *key, interpolate_plan plan)
{
  plan_cache_entry_t **entry_ptr = &cache->first;

  while(*entry_ptr != NULL && plan_compare_key(key, &(*entry_ptr)->key) != 0)
    entry_ptr =  &(*entry_ptr)->next;

  if (*entry_ptr == NULL)
  {
    plan_cache_entry_t *new_entry = malloc(sizeof(plan_cache_entry_t));

    if (new_entry == NULL)
      return 0;

    copy_key(key, &new_entry->key);
    new_entry->value = plan;
    new_entry->next = NULL;
    *entry_ptr = new_entry;
    return 1;
  }
  else
  {
    return 0;
  }
}

interpolate_plan plan_cache_get(plan_cache_t *cache, const plan_key_t *key)
{
  plan_cache_entry_t **entry_ptr = &cache->first;

  while(*entry_ptr != NULL && plan_compare_key(key, &(*entry_ptr)->key) != 0)
    entry_ptr =  &(*entry_ptr)->next;

  if (*entry_ptr == NULL)
    return NULL;
  else
    return (*entry_ptr)->value;
}
