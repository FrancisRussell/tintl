#include "tintl/allocation.h"
#include "tintl/forward.h"
#include <stdlib.h>
#include <stdint.h>

enum
{
  /// SSE byte alignment.
  ALIGNMENT = 1 << 4
};


/// Allocate SSE-aligned data.
/// @param size number of bytes.
static void *rs_alloc(size_t size)
{
  char *const data = malloc(size + sizeof(size_t) + ALIGNMENT - 1);

  if (data == NULL)
    return NULL;

  char *start = data + sizeof(size_t);
  const uintptr_t start_alignment = (uintptr_t) start & (ALIGNMENT - 1);

  if (start_alignment != 0)
    start += (ALIGNMENT - start_alignment);

  const size_t offset = start - data;
  ((size_t*) start)[-1] = offset;

  return start;
}

double *tintl_alloc_real(size_t size)
{
  return (double*) rs_alloc(size * sizeof(double));
}

rs_complex *tintl_alloc_complex(size_t size)
{
  return (rs_complex*) rs_alloc(size * sizeof(rs_complex));
}

void tintl_free(void *data)
{
  if (data == NULL)
    return;

  const size_t offset = ((size_t*) data)[-1];
  free((char*)data - offset);
}
