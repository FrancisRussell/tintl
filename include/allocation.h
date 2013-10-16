#ifndef ALLOCATION_H
#define ALLOCATION_H

/// \file
/// Aligned emory allocation routines.

#include "forward.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// Allocates a 16-byte aligned region of doubles.
/// @param size number of elements
double *rs_alloc_real(size_t size);

/// Allocates a 16-byte aligned region of complex doubles.
/// @param size number of elements
rs_complex *rs_alloc_complex(size_t size);


/// Frees memory allocated with either rs_alloc_real or rs_alloc_complex.
/// @param data the region to free
void rs_free(void *data);

#ifdef __cplusplus
}
#endif

#endif
