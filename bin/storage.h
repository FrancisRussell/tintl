#ifndef STORAGE_H
#define STORAGE_H

#include <complex.h>
#include <fftw3.h>
#include <interpolate.h>

typedef enum
{
  INTERLEAVED,
  SPLIT
} storage_layout_t;

typedef struct
{
  storage_layout_t layout;
  fftw_complex *interleaved;
  size_t num_elements;

  struct
  {
    double *real;
    double *imag;
  } split;
} storage_t;

const char *layout_name(const storage_layout_t layout);
void storage_allocate(storage_t *storage, storage_layout_t layout, size_t size);
void storage_free(storage_t *storage);
void storage_set_elem(storage_t *storage, size_t offset, fftw_complex value);
void storage_zero(storage_t *storage);
void execute_interpolate(interpolate_plan plan, storage_t *in, storage_t *out);
fftw_complex storage_get_elem(const storage_t *storage, size_t offset);

#endif
