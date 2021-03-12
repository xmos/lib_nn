#include "AggregateFn.hpp"

extern "C" {
  #include "vpu_sim.h"
}

#include <iostream>
#include <stdio.h>


int8_t* MatMulFn::boggle(int8_t *raw_weights, std::array<int, 4> &shape, int bits_per_element) {

  int element_count = bits_per_element;
  for (auto s : shape)
    element_count *= s;
  element_count /= 8; //8 bits per byte

  int kernel_size = get_kernel_size(element_count / shape[0], shape[0]);

  int8_t * boggled_weights = new int8_t [kernel_size]; 
  //TODO 


  return boggled_weights;
}

int MatMulFn :: get_scratch_size(int input_bytes) {
  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  return ((input_bytes + vpu_bytes - 1 ) / vpu_bytes ) * vpu_bytes;
}

/*
input_bytes is the number of bytes a single output channel of the kernel requires 
output_channel_count obvs
*/
int MatMulFn::get_kernel_size(int input_bytes, int output_channel_count) {

  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int vpu_ring_buffer_length = VPU_INT16_EPV; 

  int complete_channel_groups = output_channel_count / vpu_ring_buffer_length;

  int kernel_bytes = complete_channel_groups * input_bytes * vpu_ring_buffer_length;
  
  int kernel_tail = (input_bytes%vpu_bytes) * (output_channel_count - 1) + vpu_bytes;

  //for all but the last full kernel vpu word load we need vpu_bytes * output_channel_count
  int full_kernel_vpu_work_loads = input_bytes / vpu_bytes;
  if (full_kernel_vpu_work_loads > 0)
    kernel_bytes += output_channel_count * vpu_bytes * (full_kernel_vpu_work_loads-1);

  //the final load can go over
  kernel_bytes += std::max(vpu_bytes * vpu_ring_buffer_length, vpu_bytes * output_channel_count + kernel_tail);

  return kernel_bytes;
}

MatMulFn::MatMulFn(int output_slice_channel_count, size_t bytes_per_kernel_channel, int8_t * weights): 
  output_slice_channel_count(output_slice_channel_count),
  bytes_per_kernel_channel(bytes_per_kernel_channel),
  weights(weights) 
{
  //maybe compute k_p_adjust and input_channel_group_count
}

void MatMulFn::mat_mul_impl(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group){
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int vpu_epv = VPU_INT16_EPV;
  
  int32_t cur_output_channels_in_scope; //TODO  -make this a one liner
  if ((output_channel_group+1) * vpu_epv < output_slice_channel_count)
    cur_output_channels_in_scope = vpu_epv;
  else
    cur_output_channels_in_scope = output_slice_channel_count - output_channel_group * vpu_epv;

  int8_t * K_p = weights + bytes_per_kernel_channel * vpu_epv * output_channel_group;//changes

  assert(cur_output_channels_in_scope > 0);

  //These are a function of the number of output channels in scope
  //They might change depending on which output channel slice we are doing
  int step = (cur_output_channels_in_scope - (vpu_epv - 1)) * vpu_bytes; 

  int tail_loops = vpu_epv - 1 + step/vpu_bytes;

  //these are a funciton of the number of input bytes
  //These are unchanging and could be hoisted into the constructor
  int32_t k_p_adjust = bytes_per_kernel_channel%vpu_bytes;  //TODO -make this a one liner

  //The tail loop must execute in order to rotate the ring buffer to leave the results in 
  //0->N-1 of the ring buffer
  if ( k_p_adjust == 0 )
    k_p_adjust = vpu_bytes;

  int input_channel_group_count = (bytes_per_kernel_channel - k_p_adjust) / vpu_bytes; 

  int8_t * D_p = T;

  for (auto p = 0; p < input_channel_group_count; ++p){
    VLDC(vpu, D_p);

    D_p += XS3_VPU_VREG_WIDTH_BYTES;

    for(auto l = 0; l<VPU_INT16_EPV-1; l++){
      VLMACCR(vpu, K_p);
      K_p += XS3_VPU_VREG_WIDTH_BYTES;
    }
    VLMACCR(vpu, K_p);
    K_p += step;
  }

  VLDC(vpu, D_p);

  //Note: This forces kernels to be padded to word aligned boundaries 
  for(auto l=0; l< tail_loops; l++){
    VLMACCR(vpu, K_p);
    K_p += k_p_adjust;
  }

  VSTR(vpu, &A->vD);
  VSTD(vpu, &A->vR);
}

void MatMulFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group)
{
#ifdef NN_USE_REF
  mat_mul_impl(A, T, output_channel_group);
#else
  mat_mul_impl_asm(A, T, output_channel_group);
#endif // NN_USE_REF
}

MatMulDirectFn::MatMulDirectFn(ImageParams &X, WindowGeometry &K, int8_t * weights): 
  weights(weights)
{

  k_height_loop_counter = K.shape.height - 1;
  k_width_loop_counter = K.shape.width - 1;

  input_channel_loop_counter =
      (X.channels / XS3_VPU_VREG_WIDTH_BYTES) - 1;
  
  bytes_per_kernel_channel = K.shape.height * K.shape.width * X.channels;

  int bytes_per_input_channel = X.channels;

  inner_x_h_step = bytes_per_input_channel * (K.dilation.horizontal - 1);

  inner_x_v_step =
      (bytes_per_input_channel * ((X.width*K.dilation.vertical - K.shape.width))) 
        - inner_x_h_step;

}

void MatMulDirectFn::mat_mul_direct_impl(vpu_ring_buffer_t * A , int8_t * X, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  VCLRDR(vpu);

  int8_t * X_cur_p = X;

  int8_t * K_p = weights + bytes_per_kernel_channel * VPU_INT16_EPV * output_channel_group;
  for (int kh =  k_height_loop_counter; kh >= 0 ; kh-- )  {
    for (int kw = k_width_loop_counter; kw >= 0 ; kw-- )  {

      for (int ic = input_channel_loop_counter; ic >= 0 ; ic-- ) {
        VLDC(vpu, X_cur_p);

        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        for(unsigned l=0; l<VPU_INT16_EPV; l++){
          VLMACCR(vpu, K_p);
          K_p += XS3_VPU_VREG_WIDTH_BYTES;
        }
      }
      X_cur_p += inner_x_h_step;
    }
    X_cur_p += inner_x_v_step;
  }

  //save off the accumulator
  VSTR(vpu, &A->vD);
  VSTD(vpu, &A->vR);
}

void MatMulDirectFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group)
{
#ifdef NN_USE_REF
  mat_mul_direct_impl(A, T, output_channel_group);
#else
  mat_mul_direct_impl_asm(A, T, output_channel_group);
#endif // NN_USE_REF
}
