#include <cstdint>
#include <cstring>

#include "vpu.hpp"

class AggregateFn {
  public:
    virtual int8_t * aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_height, 
      int32_t output_width, int32_t output_channel_group) = 0;
};

class MatMulFn : public AggregateFn{

  int32_t output_slice_channel_count;//same for all
  int32_t elements_per_kernel_channel_group; //same for all
  int8_t * weights;

  public:
    virtual int8_t * aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_height, 
      int32_t output_width, int32_t output_channel_group);
};

class MatMulDirectFn : public AggregateFn {

  int32_t k_height_loop_counter;
  int32_t k_width_loop_counter;
  int32_t input_channel_loop_counter;
  int32_t inner_x_h_step;
  int32_t k_h_step;
  int32_t inner_x_v_step;
  int32_t k_v_step;


  public:
    virtual int8_t * aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_height, 
      int32_t output_width, int32_t output_channel_group);
};
