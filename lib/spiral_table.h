#ifndef SPIRAL_TABLE_H
#define SPIRAL_TABLE_H

#include <stddef.h>

typedef void (*spiral_interpolate_function_t)(double *, double*);

typedef struct
{
  int x;
  int y;
  int z;
  void (*initialise)(void);
  spiral_interpolate_function_t interpolate;
} spiral_function_info_t;

extern size_t spiral_function_table_size;
extern spiral_function_info_t spiral_function_table[];

#endif
