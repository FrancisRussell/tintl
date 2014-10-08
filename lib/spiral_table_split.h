#ifndef SPIRAL_TABLE_SPLIT_H
#define SPIRAL_TABLE_SPLIT_H

#include <stddef.h>

typedef void (*spiral_interpolate_function_split_t)(double *, double*, double*, double*);

typedef struct
{
  int x;
  int y;
  int z;
  void (*initialise)(void);
  spiral_interpolate_function_split_t interpolate;
  void (*cleanup)(void);
} spiral_function_info_split_t;

extern const size_t spiral_function_table_split_size;
extern spiral_function_info_split_t spiral_function_table_split[];

#endif
