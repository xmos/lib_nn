#include <limits>

#include "AggregateFn.hpp"
#include "vpu_sim.h"

using namespace nn;
// TODO: [astew] CHAR_BIT not defined if I build with Cygwin or gcc+Ubuntu+WSL.
// Not in <limits> either. [asj] this should be in limits.h
#ifndef CHAR_BIT
#define CHAR_BIT (sizeof(char) * 8)
#endif

Conv2dReorderedWeights MatMulDirectFn_DW::reorder_kernel_weights(
    int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element,
    int8_t pad_value) {
  int k_height = shape[1];
  int k_width = shape[2];
  int input_channel_count = shape[3];
  int output_channel_count = input_channel_count;

  // As it's a depthwise
  assert(shape[0] == 1);

  // if this were not true then there would be a mis-aligned load
  assert((input_channel_count % 4) == 0);

  Conv2dReorderedWeights reordered_weights(output_channel_count);

  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;

  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  int complete_channel_groups = output_channel_count / vpu_ring_buffer_length;

  // Non-complete output channel groups
  int remaining_output_channels =
      output_channel_count - (complete_channel_groups * vpu_ring_buffer_length);

  // The number of bytes in the kernel for each output channel
  int bytes_per_output_channel = (k_height * k_width * bits_per_element) / 8;

  int kernel_size =
      get_weights_bytes(bytes_per_output_channel, output_channel_count);

  // raw_weights[0][height][width][channels]

  for (int ocg = 0; ocg < complete_channel_groups; ++ocg) {
    for (int h = 0; h < k_height; ++h) {
      for (int w = 0; w < k_width; ++w) {
        // copy(raw_weights[h][w][ocg * 16], 16);
        int8_t *src = raw_weights + ocg * 16 + (w * output_channel_count) +
                      h * (output_channel_count * k_width);
        reordered_weights.weights.insert(reordered_weights.weights.end(), src,
                                         src + 16);
      }
    }
  }

  if (remaining_output_channels) {
    for (int h = 0; h < k_height; ++h) {
      for (int w = 0; w < k_width; ++w) {
        int8_t *src = raw_weights + complete_channel_groups * 16 +
                      (w * output_channel_count) +
                      h * (output_channel_count * k_width);

        reordered_weights.weights.insert(reordered_weights.weights.end(), src,
                                         src + remaining_output_channels);
        int pad_len = 16 - remaining_output_channels;
        reordered_weights.weights.resize(
            reordered_weights.weights.size() + pad_len, pad_value);
      }
    }
  }

  assert(kernel_size == reordered_weights.weights.size() + 16);

  // finally, pad with the required amount to ensure no reading of bad data
  reordered_weights.weights.resize(kernel_size, pad_value);

  return reordered_weights;
}

/*
input_bytes is the number of bytes a single output channel of the kernel
requires, i.e. kernel height x kernel width x 1.

output_channel_count obvs.
*/
int MatMulDirectFn_DW::get_weights_bytes(int input_bytes,
                                         int output_channel_count) {
  // TODO consider these defines
  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int vpu_ring_buffer_length = VPU_INT16_EPV;
  // VPU_INT16_VLMACC_ELMS

  // Count of channel groups with all 16 channels in use.
  int complete_output_channel_groups =
      output_channel_count / vpu_ring_buffer_length;

  // Count of remaining output channels after all complete output channels
  // groups have been accounted for.
  int remaining_output_channels =
      output_channel_count -
      (complete_output_channel_groups * vpu_ring_buffer_length);

  int kernel_bytes =
      complete_output_channel_groups * input_bytes * vpu_ring_buffer_length;

  if (remaining_output_channels) {
    // If there are any remaining then they will be padded to the
    // same length as the others(16).
    kernel_bytes += (input_bytes * vpu_ring_buffer_length);
  }

  // This accounts for the kernel overread, it should pad by 16 bytes.
  kernel_bytes += (vpu_bytes - vpu_ring_buffer_length);

  return kernel_bytes;
}
MatMulDirectFn_DW::Params::Params(const ImageGeometry &X,
                                  const WindowGeometry &K, int8_t *weights,
                                  int weights_bytes) {
  this->weights_bytes = weights_bytes;
  this->weights = weights;

  k_height_loop_counter = K.shape.height - 1;
  k_width_loop_counter = K.shape.width - 1;

  bytes_per_kernel_channel_group =
      K.shape.height * K.shape.width * VPU_INT16_VLMACC_ELMS;

  int bytes_per_pixel = X.PixelBytes();

  inner_x_h_step = bytes_per_pixel * K.dilation.col;
  inner_x_v_step = bytes_per_pixel * (int)X.width * (int)K.dilation.row -
                   (int)K.shape.width * bytes_per_pixel * (int)K.dilation.col;
}

void mat_mul_direct_dw_impl(MatMulDirectFn_DW::Params *params, VPURingBuffer *A,
                            int8_t *X, int32_t output_channel_group) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  int8_t *X_cur_p = X + 16 * output_channel_group;
  int8_t *K_p = (int8_t *)params->weights +
                params->bytes_per_kernel_channel_group * output_channel_group;

  for (int kh = params->k_height_loop_counter; kh >= 0; kh--) {
    for (int kw = params->k_width_loop_counter; kw >= 0; kw--) {
      VLDC(vpu, X_cur_p);
      VLMACC(vpu, K_p);
      K_p += VPU_INT16_VLMACC_ELMS;

      X_cur_p += params->inner_x_h_step;
    }
    X_cur_p += params->inner_x_v_step;
  }

  // save off the accumulator
  VSTR(vpu, &A->vR);
  VSTD(vpu, &A->vD);
}

C_API void mat_mul_direct_dw_impl_asm(MatMulDirectFn_DW::Params *params,
                                      VPURingBuffer *A, int8_t *X,
                                      int32_t output_channel_group);

void MatMulDirectFn_DW::aggregate_fn(VPURingBuffer *A, int8_t *T,
                                     int32_t output_channel_group) {
#ifdef NN_USE_REF
  mat_mul_direct_dw_impl(this->params, A, T, output_channel_group);
#else
  mat_mul_direct_dw_impl_asm(this->params, A, T, output_channel_group);
#endif  // NN_USE_REF
}
