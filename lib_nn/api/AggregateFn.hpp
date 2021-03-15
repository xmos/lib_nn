#include <cstdint>
#include <cstring>
#include <array>

#include "vpu.hpp"
#include "Image.hpp"

// template <class T>
class AggregateFn {
  public:
    virtual void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group) = 0;
    // int8_t * inline aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group) {
    //   static_cast<T*>(this)->aggregate_fn(...);
    // }
};

// these should inherit from Conv2DAggrFn
// Conv2DAggrFn class should have a boggle method

struct Conv2dReorderedWeights{
  int8_t * weights;
  int weights_byte_count;
  int8_t * final_vpu_load_addresses;
};

class MatMulFn : public AggregateFn {

  private:
  void mat_mul_impl(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);
  int8_t * weights;
  int32_t output_slice_channel_count;
  size_t bytes_per_kernel_channel;

  public:
    MatMulFn(int output_slice_channel_count, size_t bytes_per_kernel_channel, int8_t * weights);
    void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

    static int8_t* reorder_kernel_weights(int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element, int8_t pad_value);
    static int get_kernel_size(int input_bytes, int output_channel_count);
    static int get_scratch_size(int input_bytes) ;
};

class MatMulDirectFn : public AggregateFn {
  private:

  void mat_mul_direct_impl(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

  int8_t * weights;

  int32_t bytes_per_kernel_channel;

  int32_t k_height_loop_counter;
  int32_t k_width_loop_counter;
  int32_t input_channel_loop_counter;

  int32_t inner_x_h_step;
  int32_t inner_x_v_step;

  public:
    MatMulDirectFn(ImageParams &X, WindowGeometry &K, int8_t * weights);
    void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

};
