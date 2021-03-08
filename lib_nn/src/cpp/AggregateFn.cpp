#include "AggregateFn.hpp"

#include "vpu_sim.h"

int8_t * MatMulFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_height, 
      int32_t output_width, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;
  VCLRDR(vpu);
  int32_t cur_output_channels_in_scope = (output_slice_channel_count - (output_channel_group * 16) ) %16;

  int8_t * K_p = weights + elements_per_kernel_channel_group* output_channel_group;
  int32_t step = (cur_output_channels_in_scope - (16 - 1)) * 32;
  
  int32_t elements_per_kernel_channel = elements_per_kernel_channel;

  int32_t input_elements_per_output_channel ;
  int32_t k_p_adjust = input_elements_per_output_channel * cur_output_channels_in_scope; 
  int32_t input_channel_group_count;

  int8_t * D_p = T;
  
  for (int32_t p = input_channel_group_count; p > 0; p--){
    VLDC(vpu, D_p);

    D_p += XS3_VPU_VREG_WIDTH_BYTES;

    for(unsigned l=0; l<VPU_INT16_EPV-1; l++){
      VLMACCR(vpu, K_p);
      K_p += XS3_VPU_VREG_WIDTH_BYTES;
    }
    VLMACCR(vpu, K_p);
    K_p += step;
  }

  VLDC(vpu, D_p);
  //This forces kernels to be padded to word aligned boundaries *************************************
  unsigned tail_loops = VPU_INT16_EPV - 1 + step/XS3_VPU_VREG_WIDTH_BYTES;
  for(unsigned l=0; l< tail_loops; l++){
    VLMACCR(vpu, K_p);
    K_p += k_p_adjust;
  }

  //TODO save off the accumulator
  VSTR(vpu, &A[0]);
  VSTD(vpu, &A[16]);

}

int8_t * MatMulDirectFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * X, int32_t output_height, 
      int32_t output_width, int32_t output_channel_group)
{
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VCLRDR(vpu);

  int8_t * X_cur_p = X;//TODO translate X to X_p using output_height, output_width, output_channel_group

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