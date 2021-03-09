#include <cstdint>
#include <cstring>

#include "vpu.hpp"
#include "Image.hpp"

// template <class T>
class AggregateFn {
  public:
    virtual int8_t * aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group) = 0;
    // int8_t * inline aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group) {
    //   static_cast<T*>(this)->aggregate_fn(...);
    // }
};

// these should inherit from Conv2DAggrFn
// Conv2DAggrFn class should have a boggle method

class MatMulFn : public AggregateFn {

  int8_t * weights;
  int32_t output_slice_channel_count;
  size_t bytes_per_kernel_channel;

  public:
    MatMulFn(int output_slice_channel_count, size_t bytes_per_kernel_channel, int8_t * weights);
    int8_t * aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);
};

class MatMulDirectFn : public AggregateFn {

  int8_t * weights;

  int32_t k_height_loop_counter;
  int32_t k_width_loop_counter;
  int32_t input_channel_loop_counter;

  int32_t inner_x_h_step;
  int32_t inner_x_v_step;

  int32_t k_h_step;
  int32_t k_v_step;

  public:
    MatMulDirectFn(ImageParams &X, WindowGeometry &K);
    int8_t * aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

};
