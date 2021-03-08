#include "OutputTransformFn.hpp"
#include "vpu_sim.h"

int8_t * OutputT_Int8::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_chan)
{

  return Y;
}

int8_t * OutputT_Binary::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_chan)
{


  return Y;
}