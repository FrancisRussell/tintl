#ifndef TINTL_ALLOCATION_H
#define TINTL_ALLOCATION_H

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
double *tintl_alloc_real(size_t size);

/// Allocates a 16-byte aligned region of complex doubles.
/// @param size number of elements
rs_complex *tintl_alloc_complex(size_t size);


/// Frees memory allocated with either tintl_alloc_real or tintl_alloc_complex.
/// @param data the region to free
void tintl_free(void *data);

#ifdef __cplusplus
}
#endif

#endif
