#include <cuComplex.h>

#ifdef __cplusplus
extern "C"
{
#endif

void halve_nyquist_components_cuda(interpolate_properties_t *props, block_info_t *block_info, cuDoubleComplex *coarse);

void pad_coarse_to_fine_interleaved_cuda(interpolate_properties_t *props,
  const block_info_t *from_info, const cuDoubleComplex *from,
  const block_info_t *to_info, cuDoubleComplex *to,
  const int positive_only);

#ifdef __cplusplus
}
#endif
