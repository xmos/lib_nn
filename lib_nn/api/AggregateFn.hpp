#include <cstdint>
#include <cstring>
#include <array>
#include <vector>

#include "vpu.hpp"

#include "../src/cpp/filt2d/geom/ImageGeometry.hpp"
#include "../src/cpp/filt2d/geom/WindowGeometry.hpp"

namespace nn {
namespace filt2d {

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

struct Conv2dReorderedWeights {
  std::vector<int8_t> weights;
  std::vector<int> final_vpu_load_addresses;
  Conv2dReorderedWeights(int channels):
    weights(), 
    final_vpu_load_addresses(channels, 0)
  {

  }
};

class MatMulInt8 : public AggregateFn {

  public:
  class Params {
    public:
    const int8_t * weights;
    const int32_t output_slice_channel_count;
    const size_t bytes_per_kernel_channel;

    Params(const int output_slice_channel_count, const size_t bytes_per_kernel_channel, const int8_t * weights);

  };
  Params *params;
  private:
  void mat_mul_impl(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);
  
  public:
    MatMulInt8(Params *params):params(params){};
    void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

    static Conv2dReorderedWeights reorder_kernel_weights(
      int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element, int8_t pad_value);
      
    static int get_kernel_size(int input_bytes, int output_channel_count);
    static int get_scratch_size(int input_bytes) ;
};


// class MatMulBinary : public AggregateFn {

//   private:
//   void mat_mul_impl(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);
//   int8_t * weights;
//   int32_t output_slice_channel_count;
//   size_t bytes_per_kernel_channel;

//   public:
//     MatMulBinary(int output_slice_channel_count, size_t bytes_per_kernel_channel, int8_t * weights);
//     void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

//     static Conv2dReorderedWeights reorder_kernel_weights(
//       int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element, int8_t pad_value);
      
//     static int get_kernel_size(int input_bytes, int output_channel_count);
//     static int get_scratch_size(int input_bytes) ;
// };

class MatMulDirectFn : public AggregateFn {

  public:
  class Params {
    public:

    const int8_t * weights;

    int32_t bytes_per_kernel_channel;

    int32_t k_height_loop_counter;
    int32_t k_width_loop_counter;
    int32_t input_channel_loop_counter;

    int32_t inner_x_h_step;
    int32_t inner_x_v_step;

    Params(const ImageGeometry &X, const WindowGeometry &K, const int input_ch_per_output, const int8_t * weights);  
  };

  private:

  void mat_mul_direct_impl(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

  Params * params;

  public:
    MatMulDirectFn(Params * params):params(params){};
    void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group);

};
}
}