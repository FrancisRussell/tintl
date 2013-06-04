#include "allocation.h"
#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>
#include <stdint.h>

enum
{
  ALIGNMENT = 1 << 4
};

static void *rs_alloc(size_t size)
{
  char *const data = malloc(size + sizeof(size_t) + ALIGNMENT - 1);
  char *start = data + sizeof(size_t);
  const uintptr_t start_alignment = (uintptr_t) start & (ALIGNMENT - 1);

  if (start_alignment != 0)
    start += (ALIGNMENT - start_alignment);

  const size_t offset = start - data;
  ((size_t*) start)[-1] = offset;

  return start;
}

double *rs_alloc_real(size_t size)
{
  return (double*) rs_alloc(size * sizeof(double));
}

fftw_complex *rs_alloc_complex(size_t size)
{
  return (fftw_complex*) rs_alloc(size * sizeof(fftw_complex));
}

void rs_free(void *data)
{
  const size_t offset = ((size_t*) data)[-1];
  free((char*)data - offset);
}
