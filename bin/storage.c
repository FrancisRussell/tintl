#include "storage.h"
#include "tintl/allocation.h"
#include "tintl/interpolate.h"
#include <complex.h>
#include <fftw3.h>
#include <assert.h>
#include <string.h>

const char *layout_name(const storage_layout_t layout)
{
  switch(layout)
  {
    case INTERLEAVED:
      return "interleaved";
    case SPLIT:
        return "split";
    default:
        return "unknown";
  }
}

void storage_allocate(storage_t *storage, storage_layout_t layout, size_t size)
{
  assert(storage != NULL);
  storage->layout = layout;
  storage->num_elements = size;

  switch(layout)
  {
    case INTERLEAVED:
      storage->interleaved = tintl_alloc_complex(size);
      assert(storage->interleaved != NULL);
      break;
    case SPLIT:
      storage->split.real = tintl_alloc_real(size);
      storage->split.imag = tintl_alloc_real(size);
      assert(storage->split.real != NULL);
      assert(storage->split.imag != NULL);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", layout);
      exit(EXIT_FAILURE);
  }
}

void storage_free(storage_t *storage)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      tintl_free(storage->interleaved);
      break;
    case SPLIT:
      tintl_free(storage->split.real);
      tintl_free(storage->split.imag);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

void storage_set_elem(storage_t *storage, size_t offset, fftw_complex value)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      storage->interleaved[offset] = value;
      break;
    case SPLIT:
      storage->split.real[offset] = creal(value);
      storage->split.imag[offset] = cimag(value);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}

void storage_zero(storage_t *storage)
{
  switch(storage->layout)
  {
    case INTERLEAVED:
      memset(storage->interleaved, 0, storage->num_elements * sizeof(double) * 2);
      break;
    case SPLIT:
      memset(storage->split.real, 0, storage->num_elements * sizeof(double));
      memset(storage->split.imag, 0, storage->num_elements * sizeof(double));
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }

}

void execute_interpolate(interpolate_plan plan, storage_t *in, storage_t *out)
{
  assert(in != NULL);
  assert(out != NULL);
  assert(in->layout == out->layout);

  switch(in->layout)
  {
    case INTERLEAVED:
      interpolate_execute_interleaved(plan, in->interleaved, out->interleaved);
      break;
    case SPLIT:
      interpolate_execute_split(plan, in->split.real, in->split.imag, out->split.real, out->split.imag);
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", in->layout);
      exit(EXIT_FAILURE);
  }
}

fftw_complex storage_get_elem(const storage_t *storage, size_t offset)
{
  assert(storage != NULL);

  switch(storage->layout)
  {
    case INTERLEAVED:
      return storage->interleaved[offset];
    case SPLIT:
      return storage->split.real[offset] + storage->split.imag[offset] * I;
      break;
    default:
      fprintf(stderr, "Unknown storage type %d\n", storage->layout);
      exit(EXIT_FAILURE);
  }
}


