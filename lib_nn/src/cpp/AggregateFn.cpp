#include "AggregateFn.hpp"

#include <limits>

#include "vpu_sim.h"

using namespace nn;
// TODO: [astew] CHAR_BIT not defined if I build with Cygwin or gcc+Ubuntu+WSL.
// Not in <limits> either. [asj] this should be in limits.h
#ifndef CHAR_BIT
#define CHAR_BIT (sizeof(char) * 8)
#endif

static int8_t *deref2d(int8_t *p, int p_w, int h, int w) {
  return p + h * p_w + w;
}

static int16_t *deref2d(int16_t *p, int p_w, int h, int w) {
  return p + h * p_w + w;
}

Conv2dReorderedWeights MatMulBase::reorder_kernel_weights(
    int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element,
    int8_t pad_value) {
  const int vpu_ring_buffer_length = VPU_INT16_EPV;
  const int vpu_bytes_per_word = XS3_VPU_VREG_WIDTH_BYTES;

  int output_channel_count = shape[0];

  Conv2dReorderedWeights reordered_weights(output_channel_count);

  // The number of bytes in the kernel for each output channel
  int bytes_per_output_channel =
      (shape[1] * shape[2] * shape[3] * bits_per_element) / CHAR_BIT;

  int kernel_size =
      get_weights_bytes(bytes_per_output_channel, output_channel_count);

  assert(bytes_per_output_channel * output_channel_count <=
         kernel_size + vpu_ring_buffer_length * vpu_bytes_per_word);

  // For each output channel keep a record of the final vpu load
  // so the overlap betweek the desired channel and the next can
  // be accounted for.

  // The numberof output channel groups needed to compute the whole conv.
  // This is rounded up.
  int output_channel_groups =
      (output_channel_count + vpu_ring_buffer_length - 1) /
      vpu_ring_buffer_length;

  int dst_offset = 0;

  for (int ocg = 0; ocg < output_channel_groups; ++ocg) {
    int output_channels_per_ocg =
        std::min(output_channel_count - ocg * vpu_ring_buffer_length,
                 vpu_ring_buffer_length);

    int input_channel_groups =
        (bytes_per_output_channel + vpu_bytes_per_word - 1) /
        vpu_bytes_per_word;

    for (int icg = 0; icg < input_channel_groups; ++icg) {
      int ocg_offset = ocg * vpu_ring_buffer_length;

      for (int out_ch = 0; out_ch < output_channels_per_ocg; ++out_ch) {
        int bytes_in_this_vpu_copy =
            std::min(bytes_per_output_channel - icg * vpu_bytes_per_word,
                     vpu_bytes_per_word);

        // reverse order of output channels
        int reversed_out_ch = output_channels_per_ocg - 1 - out_ch;
        int8_t *src =
            deref2d(raw_weights, bytes_per_output_channel,
                    ocg_offset + reversed_out_ch, vpu_bytes_per_word * icg);

        reordered_weights.weights.insert(reordered_weights.weights.end(), src,
                                         src + bytes_in_this_vpu_copy);

        if (icg == input_channel_groups - 1)
          reordered_weights
              .final_vpu_load_addresses[ocg_offset + reversed_out_ch] =
              dst_offset;

        dst_offset += bytes_in_this_vpu_copy;
      }
    }
  }
  assert(dst_offset <= kernel_size);

  reordered_weights.weights.resize(kernel_size, pad_value);
  return reordered_weights;
}


Conv2dReorderedWeights16 MatMulBase::reorder_kernel_weights_int16(
    int16_t *raw_weights, std::array<int, 4> &shape,
    int16_t pad_value) {
  const int vpu_ring_buffer_length = VPU_INT16_EPV;
  const int vpu_bytes_per_word = XS3_VPU_VREG_WIDTH_BYTES;
  const  int bits_per_element = 16;

  int output_channel_count = shape[0];

  Conv2dReorderedWeights16 reordered_weights(output_channel_count);

  // The number of bytes in the kernel for each output channel
  int bytes_per_output_channel =
      (shape[1] * shape[2] * shape[3] * bits_per_element) / CHAR_BIT;

  int kernel_size =
      get_weights_bytes(bytes_per_output_channel, output_channel_count);

  // This is necessary because whne adding an element at a time
  // It keeps reallocating and freeing, and because it just desn't fit it keeps
  // moving up in memory until it runs out

  reordered_weights.weights.resize(shape[0] * shape[1] * shape[2] * shape[3]);
  reordered_weights.weights.resize(0);

  assert(bytes_per_output_channel * output_channel_count <=
         kernel_size + vpu_ring_buffer_length * vpu_bytes_per_word);

  // For each output channel keep a record of the final vpu load
  // so the overlap betweek the desired channel and the next can
  // be accounted for.

  // The numberof output channel groups needed to compute the whole conv.
  // This is rounded up.
  int output_channel_groups =
      (output_channel_count + vpu_ring_buffer_length - 1) /
      vpu_ring_buffer_length;

  int dst_offset = 0;

  for (int ocg = 0; ocg < output_channel_groups; ++ocg) {
    int output_channels_per_ocg =
        std::min(output_channel_count - ocg * vpu_ring_buffer_length,
                 vpu_ring_buffer_length);

    int input_channel_groups =
        (bytes_per_output_channel + vpu_bytes_per_word - 1) /
        vpu_bytes_per_word;

    for (int icg = 0; icg < input_channel_groups; ++icg) {
      int ocg_offset = ocg * vpu_ring_buffer_length;

      for (int out_ch = 0; out_ch < output_channels_per_ocg; ++out_ch) {
        int bytes_in_this_vpu_copy =
            std::min(bytes_per_output_channel - icg * vpu_bytes_per_word,
                     vpu_bytes_per_word);

        // flip 1 and 2; for 16-bit output transform
        int flipped_out_ch = out_ch;
        if ((out_ch & 3) == 1 || (out_ch & 3) == 2) {
          flipped_out_ch ^= 6;
        }
        // reverse order of output channels - for VLMACCR
        int reversed_out_ch = output_channels_per_ocg - 1 - flipped_out_ch;
        int16_t *src =
            deref2d(raw_weights, bytes_per_output_channel/2,
                    ocg_offset + reversed_out_ch, vpu_bytes_per_word/2 * icg);

        reordered_weights.weights.insert(reordered_weights.weights.end(), src,
                                         src + bytes_in_this_vpu_copy/2);

        if (icg == input_channel_groups - 1)
          reordered_weights
              .final_vpu_load_addresses[ocg_offset + reversed_out_ch] =
              dst_offset;

        dst_offset += bytes_in_this_vpu_copy;
      }
    }
  }
  assert(dst_offset <= kernel_size);

  reordered_weights.weights.resize(kernel_size, pad_value);
  return reordered_weights;
}

int MatMulBase::get_scratch_mem_bytes(int input_bytes) {
  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  return ((input_bytes + vpu_bytes - 1) / vpu_bytes) * vpu_bytes;
}

/*
input_bytes is the number of bytes a single output channel of the kernel
requires output_channel_count obvs
*/
int MatMulBase::get_weights_bytes(int bytes_per_output_channel,
                                  int output_channel_count) {
  const int vpu_bytes_per_word = XS3_VPU_VREG_WIDTH_BYTES;
  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  int kernel_bytes = 0;
  int min_bytes = 0;

  int output_channel_groups =
      (output_channel_count + vpu_ring_buffer_length - 1) /
      vpu_ring_buffer_length;

  for (int ocg = 0; ocg < output_channel_groups; ++ocg) {
    int output_channels_per_ocg =
        std::min(output_channel_count - ocg * vpu_ring_buffer_length,
                 vpu_ring_buffer_length);

    int input_channel_groups =
        (bytes_per_output_channel + vpu_bytes_per_word - 1) /
        vpu_bytes_per_word;

    for (int icg = 0; icg < input_channel_groups; ++icg) {
      min_bytes = kernel_bytes + vpu_ring_buffer_length * vpu_bytes_per_word;

      int bytes_in_this_vpu_copy =
          std::min(bytes_per_output_channel - icg * vpu_bytes_per_word,
                   vpu_bytes_per_word);

      kernel_bytes += bytes_in_this_vpu_copy * output_channels_per_ocg;
    }
  }
  return min_bytes;
}

/*
This is used for implementing int8 and binary mat mul.
*/
void mat_mul_generic_impl(const mat_mul_generic_params_t *params, VPURingBuffer *A,
                          int8_t *T, int32_t output_channel_group,
                          int8_t *weights,
                          void (*macc_inst)(xs3_vpu *vpu, const void *addr)) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  const int32_t vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int32_t vpu_epv = VPU_INT16_EPV;

  const int32_t first_output_channel = output_channel_group * vpu_epv;

  // Point K_p at the beginning of the first output channel
  int8_t *K_p = (int8_t *)weights + params->bytes_per_kernel_channel *
                                        first_output_channel;  // changes
  int step = vpu_bytes;
  int t = params->output_slice_channel_count - first_output_channel;
  t -= (vpu_epv - 1);
  if (t <= 0) step *= t;

  // these are a funciton of the number of input bytes
  // These are unchanging and could be hoisted into the constructor
  int32_t k_p_adjust = params->bytes_per_kernel_channel % vpu_bytes;
  int input_channel_group_count =
      (params->bytes_per_kernel_channel) / vpu_bytes;

  // The tail loop must execute in order to rotate the ring buffer to leave the
  // results in 0->N-1 of the ring buffer
  if (k_p_adjust == 0) {
    input_channel_group_count--;
    k_p_adjust = vpu_bytes;
  }

  int8_t *D_p = T;

  for (int p = 0; p < input_channel_group_count; ++p) {
    VLDC(vpu, D_p);

    D_p += XS3_VPU_VREG_WIDTH_BYTES;

    for (int l = 0; l < VPU_INT16_EPV - 1; l++) {
      macc_inst(vpu, K_p);
      K_p += XS3_VPU_VREG_WIDTH_BYTES;
    }
    macc_inst(vpu, K_p);
    K_p += step;
  }

  VLDC(vpu, D_p);

  int tail_loops = vpu_epv - 1 + step / vpu_bytes;

  // aligned boundaries TODO put an assert on this
  for (int l = 0; l < tail_loops; l++) {
    macc_inst(vpu, K_p);
    K_p += k_p_adjust;
  }

  VSTR(vpu, &A->vR);
  VSTD(vpu, &A->vD);
}

void mat_mul_generic_int8_impl(const mat_mul_generic_params_t *params, VPURingBuffer *A,
                               int8_t *T, int32_t output_channel_group,
                               int8_t *weights) {
  mat_mul_generic_impl(params, A, T, output_channel_group, weights, VLMACCR);
}

void mat_mul_generic_binary_impl(const mat_mul_generic_params_t *params, VPURingBuffer *A,
                                 int8_t *T, int32_t output_channel_group,
                                 int8_t *weights) {
  mat_mul_generic_impl(params, A, T, output_channel_group, weights, VLMACCR1);
}

MatMulDirectFn::MatMulDirectFn(const ImageGeometry &X, const WindowGeometry &K,
                               const int input_ch_per_output) {
  int bytes_per_copy_per_channel =
      (input_ch_per_output * X.element_bits) / CHAR_BIT;
  p.k_height_loop_counter = K.shape.height - 1;
  p.k_width_loop_counter = K.shape.width - 1;

  p.input_channel_loop_counter =
      (bytes_per_copy_per_channel / XS3_VPU_VREG_WIDTH_BYTES) - 1;

  p.bytes_per_kernel_channel = K.shape.height * K.shape.width *
                             bytes_per_copy_per_channel * VPU_INT16_EPV;

  int bytes_per_pixel = X.PixelBytes();

  assert(bytes_per_pixel == bytes_per_copy_per_channel);
  p.inner_x_h_step =
      bytes_per_pixel * K.dilation.col - bytes_per_copy_per_channel;
  p.inner_x_v_step = bytes_per_pixel * (int)X.width * (int)K.dilation.row -
                   (int)K.shape.width * bytes_per_pixel * (int)K.dilation.col;
}

void mat_mul_direct_impl(const mat_mul_direct_params_t *params, VPURingBuffer *A,
                         int8_t *X, int32_t output_channel_group,
                         int8_t *weights,
                         void (*macc_inst)(xs3_vpu *vpu, const void *addr)) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  int8_t *X_cur_p = X;

  int8_t *K_p = (int8_t *)weights +
                params->bytes_per_kernel_channel * output_channel_group;
  for (int kh = params->k_height_loop_counter; kh >= 0; kh--) {
    for (int kw = params->k_width_loop_counter; kw >= 0; kw--) {
      for (int ic = params->input_channel_loop_counter; ic >= 0; ic--) {
        VLDC(vpu, X_cur_p);

        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        for (unsigned l = 0; l < VPU_INT16_EPV; l++) {
          macc_inst(vpu, K_p);
          K_p += XS3_VPU_VREG_WIDTH_BYTES;
        }
      }
      X_cur_p += params->inner_x_h_step;
    }
    X_cur_p += params->inner_x_v_step;
  }

  // save off the accumulator
  VSTR(vpu, &A->vR);
  VSTD(vpu, &A->vD);
}

void mat_mul_direct_int8_impl(const mat_mul_direct_params_t *params, VPURingBuffer *A,
                              int8_t *X, int32_t output_channel_group,
                              int8_t *weights) {
  mat_mul_direct_impl(params, A, X, output_channel_group, weights, VLMACCR);
}

void mat_mul_direct_binary_impl(const mat_mul_direct_params_t *params,
                                VPURingBuffer *A, int8_t *X,
                                int32_t output_channel_group, int8_t *weights) {
  mat_mul_direct_impl(params, A, X, output_channel_group, weights, VLMACCR1);
}

C_API void mat_mul_direct_int16_impl_asm(const mat_mul_direct_params_t *params,
                                        VPURingBuffer *A, int16_t *X,
                                        int32_t output_channel_group,
                                        int16_t *weights);
C_API void mat_mul_direct_int8_impl_asm(const mat_mul_direct_params_t *params,
                                        VPURingBuffer *A, int8_t *X,
                                        int32_t output_channel_group,
                                        int8_t *weights);
C_API void mat_mul_generic_int8_impl_asm(const mat_mul_generic_params_t *params,
                                         VPURingBuffer *A, int8_t *X,
                                         int32_t output_channel_group,
                                         int8_t *weights);
C_API void mat_mul_direct_binary_impl_asm(const mat_mul_direct_params_t *params,
                                          VPURingBuffer *A, int8_t *X,
                                          int32_t output_channel_group,
                                          int8_t *weights);
C_API void mat_mul_generic_binary_impl_asm(const mat_mul_generic_params_t *params,
                                           VPURingBuffer *A, int8_t *X,
                                           int32_t output_channel_group,
                                           int8_t *weights);

void nn::mat_mul_direct_int8(const mat_mul_direct_params_t *params, VPURingBuffer *A, int8_t *T,
                                  int32_t output_channel_group, int8_t *weights) {
#ifdef NN_USE_REF
  mat_mul_direct_int8_impl(params, A, T, output_channel_group, weights);
#else
  mat_mul_direct_int8_impl_asm(params, A, T, output_channel_group,
                               weights);
#endif  // NN_USE_REF
}

void nn::mat_mul_direct_binary(const mat_mul_direct_params_t *params, VPURingBuffer *A, int8_t *T,
                                        int32_t output_channel_group, int8_t *weights) {
#ifdef NN_USE_REF
  mat_mul_direct_binary_impl(params, A, T, output_channel_group, weights);
#else
  mat_mul_direct_binary_impl_asm(params, A, T, output_channel_group,
                                 weights);
#endif  // NN_USE_REF
}

void nn::mat_mul_generic_int8(const mat_mul_generic_params_t *params, VPURingBuffer *A, int8_t *T,
                              int32_t output_channel_group, int8_t *weights) {
#ifdef NN_USE_REF
  mat_mul_generic_int8_impl(params, A, T, output_channel_group, weights);
#else
  mat_mul_generic_int8_impl_asm(params, A, T, output_channel_group,
                                weights);
#endif  // NN_USE_REF
}
void nn::mat_mul_generic_binary(const mat_mul_generic_params_t *params, VPURingBuffer *A, int8_t *T,
                                int32_t output_channel_group, int8_t *weights) {
#ifdef NN_USE_REF
  mat_mul_generic_binary_impl(params, A, T, output_channel_group,
                              weights);
#else
  mat_mul_generic_binary_impl_asm(params, A, T, output_channel_group,
                                  weights);
#endif  // NN_USE_REF
}

void mat_mul_direct16_impl(const mat_mul_direct_params_t *params, VPURingBuffer *A,
                           int16_t *X, int32_t output_channel_group,
                           int16_t *weights,
                           void (*macc_inst)(xs3_vpu *vpu, const void *addr)) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S16);
  VCLRDR(vpu);

  int16_t *X_cur_p = X;

  int16_t *K_p = (int16_t *)weights +
                params->bytes_per_kernel_channel * output_channel_group/2;

  for (int kh = params->k_height_loop_counter; kh >= 0; kh--) {
    for (int kw = params->k_width_loop_counter; kw >= 0; kw--) {
      for (int ic = params->input_channel_loop_counter; ic >= 0; ic--) {
        VLDC(vpu, X_cur_p);

        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES/2;

        for (unsigned l = 0; l < VPU_INT16_EPV; l++) {
          macc_inst(vpu, K_p);
          K_p += XS3_VPU_VREG_WIDTH_BYTES/2;
        }
      }
      X_cur_p += params->inner_x_h_step/2;
    }
    X_cur_p += params->inner_x_v_step/2;
  }

  // save off the accumulator
  VSTR(vpu, &A->vR);
  VSTD(vpu, &A->vD);
}

void mat_mul_direct_int16_impl(const mat_mul_direct_params_t *params,
                               VPURingBuffer *A,
                               int16_t *X, int32_t output_channel_group,
                               int16_t *weights) {
    mat_mul_direct16_impl(params, A, X, output_channel_group, weights, VLMACCR);
}

void nn::mat_mul_direct_int16(const mat_mul_direct_params_t *params,
                          VPURingBuffer *A, int16_t *T,
                          int32_t output_channel_group, int16_t *weights) {
#ifdef NN_USE_REF
    mat_mul_direct_int16_impl(params, A, T, output_channel_group, weights);
#else
    mat_mul_direct_int16_impl_asm(params, A, T, output_channel_group,
                                  weights);
#endif  // NN_USE_REF
}
