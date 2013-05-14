#include "phase_shift_interface.h"
#include "interpolate.h"

static void *plan_interpolate_3d_wrapper(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int flags)
{
  return plan_interpolate_3d(n0, n1, n2, in, out, flags);
}

static void interpolate_execute_wrapper(const void *plan, fftw_complex *in, fftw_complex *out)
{
  interpolate_execute((const interpolate_plan) plan, in, out);
}

static void interpolate_print_timings_wrapper(const void *plan)
{
  interpolate_print_timings((const interpolate_plan) plan);
}

static void interpolate_destroy_plan_wrapper(void *plan)
{
  interpolate_destroy_plan((interpolate_plan) plan);
}

static const char *interpolate_get_name(void)
{
  return "Phase shift";
}

interpolate_interface get_phase_shift_interpolate(void)
{
  interpolate_interface interface;
  interface.get_name = interpolate_get_name;
  interface.plan = plan_interpolate_3d_wrapper;
  interface.execute = interpolate_execute_wrapper;
  interface.print_timings = interpolate_print_timings_wrapper;
  interface.destroy_plan = interpolate_destroy_plan_wrapper;
  return interface;
}
