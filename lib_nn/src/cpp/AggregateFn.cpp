#include "AggregateFn.hpp"

extern "C" {
  #include "vpu_sim.h"
}

MatMulFn::MatMulFn(
  int output_slice_channel_count, 
  size_t bytes_per_kernel_channel, 
  int8_t * weights):

  output_slice_channel_count(output_slice_channel_count),
  bytes_per_kernel_channel(bytes_per_kernel_channel),
  weights(weights) {

  //maybe compute k_p_adjust and input_channel_group_count

}

foo(){
xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VCLRDR(vpu);

  const int vpu_bytes = XS3_VPU_VREG_WIDTH_BYTES;
  const int vpu_epv = VPU_INT16_EPV;
  
  int32_t cur_output_channels_in_scope = (output_slice_channel_count - (output_channel_group * vpu_epv) ) % vpu_epv;//changes
  int8_t * K_p = weights + bytes_per_kernel_channel * vpu_epv * output_channel_group;//changes

  assert(cur_output_channels_in_scope > 0);

  //These are a function of the number of output channels in scope
  //They might change depending on which output channel slice we are doing
  int step = (cur_output_channels_in_scope - (vpu_epv - 1)) * vpu_epv; 
  int tail_loops = vpu_epv - 1 + step/vpu_bytes;

  //these are a funciton of the number of input bytes
  //These are unchanging and could be hoisted into the constructor
  int32_t k_p_adjust = bytes_per_kernel_channel%vpu_bytes; 

  //The tail loop must execute in order to rotate the ring buffer to leave the results in 
  //0->N-1 of the ring buffer
  if ( k_p_adjust == 0 )
    k_p_adjust = vpu_bytes;

  int input_channel_group_count = (bytes_per_kernel_channel - k_p_adjust) / vpu_bytes; 

  int8_t * D_p = T;
  
  for (auto p = input_channel_group_count; p > 0; --p){
    VLDC(vpu, D_p);

    D_p += XS3_VPU_VREG_WIDTH_BYTES;

    for(auto l=0; l<VPU_INT16_EPV-1; l++){
      VLMACCR(vpu, K_p);
      K_p += XS3_VPU_VREG_WIDTH_BYTES;
    }
    VLMACCR(vpu, K_p);
    K_p += step;
  }

  VLDC(vpu, D_p);
  //This forces kernels to be padded to word aligned boundaries *************************************
  for(auto l=0; l< tail_loops; l++){
    VLMACCR(vpu, K_p);
    K_p += k_p_adjust;
  }

  //TODO save off the accumulator
  VSTR(vpu, &A[0]);
  VSTD(vpu, &A[16]);
}


static int8_t* MatMulFn::boggle(int8_t* raw_weights) {
  #if xs3
    static_assert<false>;
  #endif
}

int8_t * MatMulFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group)
{
  #if xs3
  foo_asm();
  #else 
  foo();
  #endif
}


MatMulDirectFn::MatMulDirectFn(ImageParams &X, WindowGeometry &K){

}

int8_t * MatMulDirectFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * X, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VCLRDR(vpu);

  int8_t * X_cur_p = X;

  int8_t * K_p = 0; //get K_p from direct_matmul_params using output_channel_group
  
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
      K_p += k_h_step;
    }
    X_cur_p += inner_x_v_step;
    K_p += k_v_step;
  }

  //save off the accumulator
  VSTR(vpu, &A[0]);
  VSTD(vpu, &A[16]);

}