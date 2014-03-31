#ifndef TINTL_PAPI_MULTIPLEX_H
#define TINTL_PAPI_MULTIPLEX_H

#include <papi.h>
#include <stddef.h>

typedef int papi_event_t;
typedef int papi_eventset_t;

#define PAPI_CHECK(err) pm_handle_papi_error(err, __FILE__, __LINE__);
void pm_handle_papi_error(const int error, const char *const file, const int line);

static const papi_event_t pm_common_events[] = {PAPI_TOT_CYC, PAPI_REF_CYC, PAPI_TOT_INS};
#define pm_num_common_events (sizeof(pm_common_events)/sizeof(pm_common_events[0]))

struct pm_event_info;

typedef struct
{
  size_t set_count;
  size_t set_index;
  long long total_samples;
  long long common_values[pm_num_common_events];
  struct pm_event_info *events_info;
} papi_multiplex_t;

int pm_init(papi_multiplex_t *pm);
int pm_add_event(papi_multiplex_t *pm, papi_event_t event);
int pm_start(papi_multiplex_t *pm);
int pm_stop(papi_multiplex_t *pm);
int pm_count(papi_multiplex_t *pm, papi_event_t event, long long *value);
int pm_destroy(papi_multiplex_t *pm);

#endif
