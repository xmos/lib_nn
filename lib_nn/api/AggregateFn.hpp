#ifndef LIB_NN_AGGREGATE_FN_HPP_
#define LIB_NN_AGGREGATE_FN_HPP_

#include <array>
#include <vector>

#include "Utils.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "vpu.hpp"
#include "xs3_vpu.h"

namespace nn {

struct Conv2dReorderedWeights {
  /**
   * @brief Byte vector of the weights in the order that the VPU can most
   * efficiently execute.
   *
   */
  std::vector<int8_t> weights;

  /**
   * @brief vector of offsets into the weights. Each of the offsets represents
   * the location that the VPU will load from in during the final load. The
   * purpose of this vector is to allow the computation of an accumulator
   * adjustment where nessessary. For example the final load might look like: |
   * N bytes of weights | 32 - N bytes of unwanted weights | where N is between
   * 1 and 32. Knowing the 32 - N bytes of unwanted weights coupled with the
   * scratch memory padding value allows the computation of a statically
   * knowable accumulator offset. This offset can then be included into the
   * bias. This allows dense packing of weights.
   */
  std::vector<int> final_vpu_load_addresses;

  Conv2dReorderedWeights(int channels)
      : weights(), final_vpu_load_addresses(channels, 0) {}
};


struct mat_mul_generic_params_t{
  int32_t output_slice_channel_count;
  int32_t bytes_per_kernel_channel;
};

class MatMulBase {
  private:
  mat_mul_generic_params_t p;
 public:
  MatMulBase(int output_slice_channel_count, int32_t bytes_per_kernel_channel)
  : p{output_slice_channel_count, bytes_per_kernel_channel} {}
  mat_mul_generic_params_t getParams() {return p;};
  /**
   * @brief Used to reorder the weights from their normal form ([OutputChannel,
   * Height, Width, InputChannel]) to a form conducive to the VPU efficiently
   * loading and multiplying them.
   *
   * @param raw_weights Pointer to the raw weights.
   * @param shape [OutputChannels, Height, Width, InputChannels]
   * @param bits_per_element The count of bits per element, i.e. int8 is 8,
   * binary is 1.
   * @param pad_value The value used for padding to achieve alignment where
   * nessessary. This is not window padding. It can effect the accumulator
   * result and must be compensated for where nessessary, i.e. padding with zero
   * will not effect an int8, int16 or int32 matmul, padding with
   * zeros(representing 1) will effect a binary matmul.
   * @return Conv2dReorderedWeights
   */
  static Conv2dReorderedWeights reorder_kernel_weights(
      int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element,
      int8_t pad_value);

  /**
   * @brief Get the required size of the weights array. This is a non-trivial
   * computation as it accounts for how the VPU will access the weights array.
   * It includes padding at after the weights, this padding exists to ensure
   * that all memory accesses of the VPU are of valid memory.
   * [asj] we could stop emiting this as OOB accesses shouldnt cause a problem
   * and only cost memory.
   *
   * @param input_bytes_per_channel The count of bytes in a single channel, i.e.
   * the product of kernel height, kernel width, input channel count and bytes
   * per element.
   * @param output_channel_count The number of output channels that these
   * weights will compute.
   * @return The size in bytes that will hold the weights and guarentee no OOB
   * accesses.
   */
  static int get_weights_bytes(int input_bytes_per_channel,
                               int output_channel_count);

  /**
   * @brief Get the required size of the scratch array. This is a non-trivial
   * computation as it accounts for how the VPU will access the scratch array.
   *
   * @param input_bytes The count of bytes in a single channel.
   * @return The size in bytes that will hold the copy of the current patch and
   * guarentee no OOB accesses.
   */
  static int get_scratch_mem_bytes(int input_bytes);
};

class MatMulInt8 : public MatMulBase {
 public:
  MatMulInt8(int output_slice_channel_count, int32_t bytes_per_kernel_channel) 
  : MatMulBase(output_slice_channel_count, bytes_per_kernel_channel){};
};

void mat_mul_generic_int8(const mat_mul_generic_params_t *params, VPURingBuffer *A, int8_t *T,
                              int32_t output_channel_group, int8_t *weights);

class MatMulBinary : public MatMulBase {
 public:
  MatMulBinary(int output_slice_channel_count, int32_t bytes_per_kernel_channel) 
  : MatMulBase(output_slice_channel_count, bytes_per_kernel_channel){};
};

void mat_mul_generic_binary(const mat_mul_generic_params_t *params, VPURingBuffer *A, int8_t *T,
                                int32_t output_channel_group, int8_t *weights);

struct mat_mul_direct_params_t{
    int32_t bytes_per_kernel_channel;
    int32_t k_height_loop_counter;
    int32_t k_width_loop_counter;
    int32_t input_channel_loop_counter;
    int32_t inner_x_h_step;
    int32_t inner_x_v_step;
};

class MatMulDirectFn {
  private:
  mat_mul_direct_params_t p;
  public:
  MatMulDirectFn(const ImageGeometry &X, const WindowGeometry &K,
                               const int input_ch_per_output);
  mat_mul_direct_params_t getParams() {return p;};
};

void mat_mul_direct_int8(const mat_mul_direct_params_t *params, VPURingBuffer *A,
                              int8_t *X, int32_t output_channel_group,
                              int8_t *weights);

class MatMulBinaryDirectFn : public MatMulDirectFn {
 public:
  MatMulBinaryDirectFn(const ImageGeometry &X, const WindowGeometry &K,
                               const int input_ch_per_output) : MatMulDirectFn(X, K, input_ch_per_output){};
};

void mat_mul_direct_binary(const mat_mul_direct_params_t *params, VPURingBuffer *A, int8_t *T,
                                        int32_t output_channel_group, int8_t *weights);

// Depthwise below here
// ////////////////////////////////////////////////////////////////////////////

struct mat_mul_dw_direct_params_t{
    /*
     The count of bytes that a channel group contains. It is used to dereference
     the weights pointer to the correct channel group start, i.e.
      int8_t *K_p = params->weights + bytes_per_kernel_channel_group *
     output_channel_group;
    */
    int32_t bytes_per_kernel_channel_group;

    int32_t k_height_loop_counter;
    int32_t k_width_loop_counter;

    int32_t inner_x_h_step;
    int32_t inner_x_v_step;
};

class MatMulDirectFn_DW {
  private:
  mat_mul_dw_direct_params_t p;
  public:
  MatMulDirectFn_DW(const ImageGeometry &X, const WindowGeometry &K);
  MatMulDirectFn_DW(const WindowGeometry &K);
  mat_mul_dw_direct_params_t getParams() {return p;};

  /**
   * @brief Used to reorder the weights from their normal form ([OutputChannel,
   * Height, Width, InputChannel]) to a form conducive to the VPU efficiently
   * loading and multiplying them.
   *
   * @param raw_weights Pointer to the raw weights.
   * @param shape [OutputChannels, Height, Width, InputChannels]
   * @param bits_per_element The count of bits per element, i.e. int8 is 8,
   * binary is 1.
   * @param pad_value The value used for padding to achieve alignment where
   * nessessary. This is not window padding. It can effect the accumulator
   * result and must be compensated for where nessessary, i.e. padding with zero
   * will not effect an int8, int16 or int32 matmul, padding with
   * zeros(representing 1) will effect a binary matmul.
   * @return Conv2dReorderedWeights
   */
  static Conv2dReorderedWeights reorder_kernel_weights(
      int8_t *raw_weights, std::array<int, 4> &shape, int8_t pad_value);

  /**
   * @brief Get the required size of the weights array. This is a non-trivial
   * computation as it accounts for how the VPU will access the weights array.
   *
   * @param input_bytes The number of bytes a single output channel of the
   kernel requires, i.e. kernel height x kernel width.
   * @param output_channel_count The number of output channels that these
   * weights will compute.
   * @return The size in bytes that will hold the weights and guarentee no OOB
   * accesses.
   */
  static int get_weights_bytes(int input_bytes, int output_channel_count);

  /**
   * @brief Get the required size of the scratch array. This is a non-trivial
   * computation as it accounts for how the VPU will access the scratch array.
   *
   * @param kernel_shape The shape of the kernel. It should look like: [1,
   * k_height, k_width, input channels]
   * @return The size in bytes that will hold the copy of the current patch and
   * guarentee no OOB accesses.
   */
  static int get_scratch_mem_bytes(std::array<int, 4> &kernel_shape);
};

void mat_mul_dw_direct(const mat_mul_dw_direct_params_t *params, VPURingBuffer *A, int8_t *T,
                                     int32_t output_channel_group, int8_t *weights);

}  // namespace nn

#endif  // LIB_NN_AGGREGATE_FN_HPP_
