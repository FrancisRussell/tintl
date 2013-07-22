#include <complex.h>
#include <stdlib.h>
#include <fftw3.h>
#include <interpolate.h>

const char *interpolate_get_name(const interpolate_plan plan)
{
  return plan->get_name(plan->detail);
}

void interpolate_execute_interleaved(const interpolate_plan plan, fftw_complex *in, fftw_complex *out)
{
  plan->execute_interleaved(plan->detail, in, out);
}

void interpolate_execute_split(const interpolate_plan plan, double *rin, double *iin, double *rout, double *iout)
{
  plan->execute_split(plan->detail, rin, iin, rout, iout);
}

void interpolate_execute_split_product(const interpolate_plan plan, const void *detail, double *rin, double *iin, double *out)
{
  plan->execute_split_product(plan->detail, rin, iin, out);
}

void interpolate_print_timings(const interpolate_plan plan)
{
  plan->print_timings(plan->detail);
}

void interpolate_destroy_plan(interpolate_plan plan)
{
  plan->destroy_detail(plan->detail);
  free(plan);
}


