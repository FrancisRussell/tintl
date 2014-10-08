#ifndef SPIRAL_TABLE_PACKED_H
#define SPIRAL_TABLE_PACKED_H

#include <stddef.h>

typedef void (*spiral_interpolate_function_packed_t)(double *, double*);

typedef struct
{
  int x;
  int y;
  int z;
  void (*initialise)(void);
  spiral_interpolate_function_packed_t interpolate;
  void (*cleanup)(void);
} spiral_function_info_packed_t;

extern const size_t spiral_function_table_packed_size;
extern spiral_function_info_packed_t spiral_function_table_packed[];

#endif
