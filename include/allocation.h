#ifndef ALLOCATION_H
#define ALLOCATION_H

#include <complex.h>
#include <fftw3.h>
#include <stdlib.h>

double *rs_alloc_real(size_t size);
fftw_complex *rs_alloc_complex(size_t size);
void rs_free(void *data);

#endif
