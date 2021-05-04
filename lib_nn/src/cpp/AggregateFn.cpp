#include <limits>

#include "AggregateFn.hpp"
#include "vpu_sim.h"

using namespace nn;
//TODO: [astew] CHAR_BIT not defined if I build with Cygwin or gcc+Ubuntu+WSL. Not in <limits> either.
//[asj] this should be in limits.h
#ifndef CHAR_BIT
#define CHAR_BIT (sizeof(char) * 8)
#endif

int8_t *deref2d(int8_t *p, int p_w, int h, int w)
{
  return p + h * p_w + w;
}

Conv2dReorderedWeights MatMulInt8::reorder_kernel_weights(int8_t *raw_weights, std::array<int, 4> &shape,
                                                          int bits_per_element, int8_t pad_value)
{

  const int vpu_ring_buffer_length = VPU_INT16_EPV;
  const int vpu_bytes_per_word = XS3_VPU_VREG_WIDTH_BYTES;

  int output_channel_count = shape[0];

  Conv2dReorderedWeights reordered_weights(output_channel_count);

  //The number of bytes in the kernel for each output channel
  int bytes_per_output_channel = (shape[1] * shape[2] * shape[3] * bits_per_element) / 8;

  int kernel_size = get_kernel_size(bytes_per_output_channel, output_channel_count);

  //For each output channel keep a record of the final vpu load
  //so the overlap betweek the desired channel and the next can
  //be accounted for.

  //The numberof output channel groups needed to compute the whole conv.
  //This is rounded up.
  int output_channel_groups =
      (output_channel_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

  int dst_offset = 0;

  for (int ocg = 0; ocg < output_channel_groups; ++ocg)
  {
    int output_channels_per_ocg = std::min(output_channel_count - ocg * vpu_ring_buffer_length, vpu_ring_buffer_length);

    int input_channel_groups = (bytes_per_output_channel + vpu_bytes_per_word - 1) / vpu_bytes_per_word;

    for (int icg = 0; icg < input_channel_groups; ++icg)
    {

      int ocg_offset = ocg * vpu_ring_buffer_length;

      for (int out_ch = 0; out_ch < output_channels_per_ocg; ++out_ch)
      {
        int bytes_in_this_vpu_copy = std::min(bytes_per_output_channel - icg * vpu_bytes_per_word, vpu_bytes_per_word);

        //reverse order of output channels
        int reversed_out_ch = output_channels_per_ocg - 1 - out_ch;
        int8_t *src = deref2d(raw_weights, bytes_per_output_channel, ocg_offset + reversed_out_ch, vpu_bytes_per_word * icg);

        reordered_weights.weights.insert(reordered_weights.weights.end(), src, src + bytes_in_this_vpu_copy);

        if (icg == input_channel_groups - 1)
          reordered_weights.final_vpu_load_addresses[ocg_offset + reversed_out_ch] = dst_offset;

        dst_offset += bytes_in_this_vpu_copy;
      }
    }
  }
  assert(dst_offset <= kernel_size);

  reordered_weights.weights.resize(kernel_size, pad_value);
  return reordered_weights;
}

int MatMulInt8 ::get_scratch_size(int input_bytes)
{
  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  return ((input_bytes + vpu_bytes - 1) / vpu_bytes) * vpu_bytes;
}

/*
input_bytes is the number of bytes a single output channel of the kernel requires 
output_channel_count obvs
*/
int MatMulInt8::get_kernel_size(int input_bytes, int output_channel_count)
{

  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  int complete_channel_groups = output_channel_count / vpu_ring_buffer_length;

  int kernel_bytes = complete_channel_groups * input_bytes * vpu_ring_buffer_length;

  int kernel_tail = (input_bytes % vpu_bytes) * (output_channel_count - 1) + vpu_bytes;

  //for all but the last full kernel vpu word load we need vpu_bytes * output_channel_count
  int full_kernel_vpu_work_loads = input_bytes / vpu_bytes;
  if (full_kernel_vpu_work_loads > 0)
    kernel_bytes += output_channel_count * vpu_bytes * (full_kernel_vpu_work_loads - 1);

  //the final load can go over
  kernel_bytes += std::max(vpu_bytes * vpu_ring_buffer_length, vpu_bytes * output_channel_count + kernel_tail);

  return kernel_bytes;
}

MatMulInt8::Params::Params(const int output_slice_channel_count, const size_t bytes_per_kernel_channel, const int8_t *weights) : weights(weights),
                                                                                                                                 output_slice_channel_count(output_slice_channel_count),
                                                                                                                                 bytes_per_kernel_channel(bytes_per_kernel_channel)
{
  //maybe compute k_p_adjust and input_channel_group_count
}

void MatMulInt8::mat_mul_impl(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  const int32_t vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int32_t vpu_epv = VPU_INT16_EPV;

  int32_t cur_output_channels_in_scope = std::min(params->output_slice_channel_count - output_channel_group * vpu_epv, vpu_epv);

  int8_t *K_p = (int8_t *)params->weights + params->bytes_per_kernel_channel * vpu_epv * output_channel_group; //changes

  assert(cur_output_channels_in_scope > 0);

  //These are a function of the number of output channels in scope
  //They might change depending on which output channel slice we are doing
  int step = (cur_output_channels_in_scope - (vpu_epv - 1)) * vpu_bytes;

  int tail_loops = vpu_epv - 1 + step / vpu_bytes;

  //these are a funciton of the number of input bytes
  //These are unchanging and could be hoisted into the constructor
  int32_t k_p_adjust = params->bytes_per_kernel_channel % vpu_bytes; //TODO -make this a one liner

  //The tail loop must execute in order to rotate the ring buffer to leave the results in
  //0->N-1 of the ring buffer
  if (k_p_adjust == 0)
    k_p_adjust = vpu_bytes;

  int input_channel_group_count = (params->bytes_per_kernel_channel - k_p_adjust) / vpu_bytes;

  // std::cout << "k_p_adjust: " << k_p_adjust << std::endl;
  // std::cout << "input_channel_group_count: " << input_channel_group_count << std::endl;

  int8_t *D_p = T;

  for (int p = 0; p < input_channel_group_count; ++p)
  {
    VLDC(vpu, D_p);

    D_p += XS3_VPU_VREG_WIDTH_BYTES;

    for (int l = 0; l < VPU_INT16_EPV - 1; l++)
    {
      VLMACCR(vpu, K_p);
      K_p += XS3_VPU_VREG_WIDTH_BYTES;
    }
    VLMACCR(vpu, K_p);
    K_p += step;
  }

  VLDC(vpu, D_p);

  //Note: This forces kernels to be padded to word aligned boundaries TODO put an assert on this
  for (int l = 0; l < tail_loops; l++)
  {
    VLMACCR(vpu, K_p);
    K_p += k_p_adjust;
  }

  VSTR(vpu, &A->vD);
  VSTD(vpu, &A->vR);
}

MatMulDirectFn::Params::Params(const ImageGeometry &X, const WindowGeometry &K, const int input_ch_per_output, const int8_t *weights) : weights(weights)
{

  //TODO X.bits_per_element
  // int bytes_per_copy_per_channel = (input_ch_per_output *  X.bits_per_element) / CHAR_BIT;
  int bytes_per_copy_per_channel = (input_ch_per_output * CHAR_BIT) / CHAR_BIT;

  k_height_loop_counter = K.shape.height - 1;
  k_width_loop_counter = K.shape.width - 1;

  input_channel_loop_counter =
      (bytes_per_copy_per_channel / XS3_VPU_VREG_WIDTH_BYTES) - 1;

  bytes_per_kernel_channel = K.shape.height * K.shape.width * X.depth;

  int bytes_per_pixel = X.pixelBytes();

  inner_x_h_step = bytes_per_pixel * K.dilation.col - bytes_per_copy_per_channel;
  inner_x_v_step = bytes_per_pixel * (int)X.width * (int)K.dilation.row - (int)K.shape.width * bytes_per_pixel * (int)K.dilation.col;
}

void MatMulDirectFn::mat_mul_direct_impl(vpu_ring_buffer_t *A, int8_t *X, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  int8_t *X_cur_p = X;

  int8_t *K_p = (int8_t *)params->weights + params->bytes_per_kernel_channel * VPU_INT16_EPV * output_channel_group;
  for (int kh = params->k_height_loop_counter; kh >= 0; kh--)
  {
    for (int kw = params->k_width_loop_counter; kw >= 0; kw--)
    {

      for (int ic = params->input_channel_loop_counter; ic >= 0; ic--)
      {
        VLDC(vpu, X_cur_p);

        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        for (unsigned l = 0; l < VPU_INT16_EPV; l++)
        {
          VLMACCR(vpu, K_p);
          K_p += XS3_VPU_VREG_WIDTH_BYTES;
        }
      }
      X_cur_p += params->inner_x_h_step;
    }
    X_cur_p += params->inner_x_v_step;
  }

  //save off the accumulator
  VSTR(vpu, &A->vD);
  VSTD(vpu, &A->vR);
}

void MatMulBinaryDirectFn::mat_mul_direct_impl(vpu_ring_buffer_t *A, int8_t *X, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  int8_t *X_cur_p = X;

  int8_t *K_p = (int8_t *)params->weights + params->bytes_per_kernel_channel * VPU_INT16_EPV * output_channel_group;
  for (int kh = params->k_height_loop_counter; kh >= 0; kh--)
  {
    for (int kw = params->k_width_loop_counter; kw >= 0; kw--)
    {

      for (int ic = params->input_channel_loop_counter; ic >= 0; ic--)
      {
        VLDC(vpu, X_cur_p);

        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        for (unsigned l = 0; l < VPU_INT16_EPV; l++)
        {
          VLMACCR1(vpu, K_p);
          K_p += XS3_VPU_VREG_WIDTH_BYTES;
        }
      }
      X_cur_p += params->inner_x_h_step;
    }
    X_cur_p += params->inner_x_v_step;
  }

  //save off the accumulator
  VSTR(vpu, &A->vD);
  VSTD(vpu, &A->vR);
}

void MatMulDirectFn::aggregate_fn(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group)
{
#ifdef NN_USE_REF
  mat_mul_direct_impl(A, T, output_channel_group);
#else
  // mat_mul_direct_impl_asm(A, T, output_channel_group);
#endif // NN_USE_REF
}

void MatMulInt8::aggregate_fn(vpu_ring_buffer_t *A, int8_t *T, int32_t output_channel_group)
{
#ifdef NN_USE_REF
  mat_mul_impl(A, T, output_channel_group);
#else
  // mat_mul_impl_asm(A, T, output_channel_group);
#endif // NN_USE_REF
}
