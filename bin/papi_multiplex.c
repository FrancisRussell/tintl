#include "papi_multiplex.h"
#include <papi.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

static int pm_add_empty_event_set(papi_multiplex_t *pm);

typedef struct pm_event_info
{
  long long samples;
  long long value;
  papi_event_t event;
  papi_eventset_t event_set;
} pm_event_info_t;

void pm_handle_papi_error(const int error, const char *const file, const int line)
{
  if (error < 0)
  {
    fprintf(stderr, "Error received from PAPI in %s at line %d: ", file, line);
    const char *const errorString = PAPI_strerror(error);

    if (errorString == NULL)
      fprintf(stderr, "Unrecognised error code (%d).\n", error);
    else
      fprintf(stderr, "%s.\n", errorString);

    exit(EXIT_FAILURE);
  }
}

static int pm_add_empty_event_set(papi_multiplex_t *pm)
{
  assert(pm != NULL);

  pm_event_info_t *new_events_info = realloc(pm->events_info, (pm->set_count + 1) * sizeof(struct pm_event_info));
  if (new_events_info == NULL)
    return PAPI_ENOMEM;

  pm->events_info = new_events_info;
  ++pm->set_count;
  papi_eventset_t *new_set = &pm->events_info[pm->set_count - 1].event_set;
  *new_set = PAPI_NULL;

  int papi_res;
  if ((papi_res = PAPI_create_eventset(new_set)) != PAPI_OK)
    return papi_res;

  if ((papi_res = PAPI_add_events(*new_set, (papi_event_t*) pm_common_events, pm_num_common_events)) != PAPI_OK)
    return papi_res;

  return PAPI_OK;
}

int pm_init(papi_multiplex_t *pm)
{
  assert(pm != NULL);

  pm->set_count = 0;
  pm->set_index = 0;
  pm->total_samples = 0;

  for(size_t i = 0; i < pm_num_common_events; ++i)
    pm->common_values[i] = 0;

  pm->events_info = NULL;

  if (!PAPI_is_initialized())
  {
    const int papi_res = PAPI_library_init(PAPI_VER_CURRENT);

    if (papi_res < 0)
      return papi_res;

    if (papi_res != PAPI_VER_CURRENT)
    {
      fprintf(stderr, "PAPI library version mismatch!");
      exit(EXIT_FAILURE);
    }
  }
  return pm_add_empty_event_set(pm);
}

int pm_add_event(papi_multiplex_t *pm, papi_event_t event)
{
  assert(pm != NULL);

  for(size_t i = 0; i < pm_num_common_events; ++i)
    if (pm_common_events[i] == event)
      return PAPI_OK;

  papi_eventset_t *final_set = &pm->events_info[pm->set_count - 1].event_set;
  if (PAPI_num_events(*final_set) > pm_num_common_events)
  {
    int papi_res = pm_add_empty_event_set(pm);
    if ((papi_res = pm_add_empty_event_set(pm)) != PAPI_OK)
      return papi_res;
  }

  final_set = &pm->events_info[pm->set_count - 1].event_set;
  pm->events_info[pm->set_count - 1].event = event;
  pm->events_info[pm->set_count - 1].samples = 0;
  pm->events_info[pm->set_count - 1].value = 0;

  int papi_res;
  if ((papi_res = PAPI_add_event(*final_set, event)) != PAPI_OK)
    return papi_res;

  return PAPI_OK;
}

int pm_start(papi_multiplex_t *pm)
{
  assert(pm != NULL);
  return PAPI_start(pm->events_info[pm->set_index].event_set);
}

int pm_stop(papi_multiplex_t *pm)
{
  assert(pm != NULL);

  const papi_eventset_t event_set = pm->events_info[pm->set_index].event_set;
  long long counts[pm_num_common_events + 1];
  int papi_res;
  if ((papi_res = PAPI_stop(event_set, counts)) != PAPI_OK)
    return papi_res;

  for(size_t i = 0; i < pm_num_common_events; ++i)
    pm->common_values[i] += counts[i];

  if (PAPI_num_events(event_set) > pm_num_common_events)
  {
    pm->events_info[pm->set_index].value += counts[pm_num_common_events];
    ++pm->events_info[pm->set_index].samples;
  }

  ++pm->total_samples;
  ++pm->set_index;

  if (pm->set_index >= pm->set_count)
    pm->set_index = 0;

  return PAPI_OK;
}

int pm_count(papi_multiplex_t *pm, const papi_event_t event, long long *value)
{
  assert(pm != NULL);

  for (size_t i = 0; i < pm_num_common_events; ++i)
  {
    if (pm_common_events[i] == event)
    {
      *value = pm->common_values[i] / pm->total_samples;
      return PAPI_OK;
    }
  }

  for(size_t i = 0; i < pm->set_count; ++i)
  {
    const papi_eventset_t event_set = pm->events_info[i].event_set;
    if (PAPI_num_events(event_set) > pm_num_common_events && pm->events_info[i].event == event)
    {
      *value = pm->events_info[i].value / pm->events_info[i].samples;
      return PAPI_OK;
    }
  }

  return PAPI_EINVAL;
}

int pm_destroy(papi_multiplex_t *pm)
{
  assert(pm != NULL);

  for(size_t i = 0; i < pm->set_count; ++i)
  {
    int papi_res;
    if ((papi_res = PAPI_cleanup_eventset(pm->events_info[i].event_set)) != PAPI_OK)
      return papi_res;

    if ((papi_res = PAPI_destroy_eventset(&pm->events_info[i].event_set)) != PAPI_OK)
      return papi_res;
  }

  free(pm->events_info);
  return PAPI_OK;
}
