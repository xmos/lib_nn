#include <algorithm>
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <cmath>
#include <tuple>
#include <limits>
#include "AggregateFn.hpp"

extern "C" {
  #include "vpu_sim.h"
}


static int clrsb(int x){
  #if defined(__XS3A__)
  for (unsigned i=0;i<32;i++){
    int y = (x<<i)>>i;
    if (y != x)
      return (i-1);
  }
  return 32;
  #else
  return __builtin_clrsb(x);
  #endif
}

// This puts upper and lower limits on the range of A
// A must reduce the vpu accumulator to 16 bit
// A must not remove all the imformation from the vpu accumulator
static void get_bounds_on_A(int* min_A, int* max_A, int32_t vpu_min_accu,
                            int32_t vpu_max_accu, int32_t vpu_clamp_min,
                            int32_t vpu_clamp_max) {
  int32_t max_out =
      std::max(std::max(std::max(vpu_min_accu, vpu_max_accu), vpu_clamp_min), vpu_clamp_max);
  int32_t min_out =
      std::min(std::min(std::min(vpu_min_accu, vpu_max_accu), vpu_clamp_min), vpu_clamp_max);
  int rsb = std::min(clrsb(max_out), clrsb(min_out));

  *max_A = rsb - 16;
  *min_A = *max_A - 16 + 1;
}

// This puts upper and lower limits on the range of Exp
// Exp will be applied to each of the values
// Exp must not saturate and of the values
// Exp must not leave all results as zero
static void get_bounds_on_Exp(int* min_Exp, int* max_Exp, float* values,
                              unsigned values_length, int bound_width) {
  assert(values_length > 0);
  int max_exponent = std::numeric_limits<int>::min();
  for (unsigned i = 0; i < values_length; i++) {
    int e;
    std::frexp(values[i], &e);
    max_exponent = std::max(max_exponent, e);
  }

  *min_Exp = -max_exponent - 1;
  *max_Exp = *min_Exp + bound_width;
}


static void solve_constraint(
    int * B_res, 
    int * A_res, 
    int * M_res,

    float* vpu_output_transform_multiplier,
    float* vpu_output_transform_bias, 
    unsigned chans_out,

    int32_t vpu_min_accu,
    int32_t vpu_max_accu, 
    int32_t vpu_clamp_min, 
    int32_t vpu_clamp_max
    ){
  int min_A, max_A;
  int min_B, max_B;
  int min_M, max_M;

  get_bounds_on_A(&min_A, &max_A, vpu_min_accu, vpu_max_accu, vpu_clamp_min, vpu_clamp_max);

  get_bounds_on_Exp(&min_M, &max_M, vpu_output_transform_multiplier, chans_out, 16);

  //This is 30 as we cannot make a 32 bit bias with a shr of 14
  get_bounds_on_Exp(&min_B, &max_B, vpu_output_transform_bias, chans_out, 16 + 14);

  // we also know that A + M = B;
  // Subtract one to ensure the addition is fine (one from A*M, B is already 30 bit at most)
  max_B = std::min(max_A + max_M - 1, max_B);
    
  // printf("min_B:%d max_B:%d\n", min_B, max_B);

  for (int A = max_A; A >= min_A; A--) {
    for (int M = max_M; M >= min_M; M--) {
      // We can squeeze a little more out of the arith by modelling
      // max_Product = max_A * max_M
      // this way we wouldnt need to subtract 2 from max_B

      int B = A + M; 

      if ((B >= min_B) && (B <= max_B)) {
        *B_res = B;
        *A_res = A;
        *M_res = M;
        return;
      }
    }
  }
  assert(0);
}

struct QuantisationParams{
  int accu_shr;
  int bias_shr;
  int multiplier_shr;
};

/*
  This is intended to handle 
*/
void quantise_activation(
    std::vector<float> & output_transform_multiplier,
    std::vector<float> & output_transform_bias, 
    int accu_min,
    int accu_max,
    std::vector<int> & chan_overlaps)

{


  // QuantisationParams q = solve_constraint(
    
  //   vpu_output_transform_multiplier,
  //   vpu_output_transform_bias, 

  //   vpu_min_accu,
  //   vpu_max_accu);

}

int8_t * deref2d(int8_t * p, int p_w, int h, int w){
  return p + h*p_w + w;
}

std::tuple<int8_t *, int8_t **, int> MatMulFn::reorder_kernel_weights(int8_t *raw_weights, std::array<int, 4> &shape, 
  int bits_per_element, int8_t pad_value) 
{

  const int vpu_ring_buffer_length = 16;
  const int vpu_bytes_per_word = 32;

  int output_channel_count = shape[0];

  Conv2dReorderedWeights reordered_weights(output_channel_count);

  //The number of bytes in the kernel for each output channel
  int bytes_per_output_channel = (shape[1]*shape[2]*shape[3]*bits_per_element)/8;

  int kernel_size = get_kernel_size(bytes_per_output_channel,output_channel_count);

  int8_t * boggled_weights = new int8_t [kernel_size]; 

  //For each output channel keep a record of the final vpu load
  //so the overlap betweek the desired channel and the next can
  //be accounted for.
  int8_t ** final_load_locations = new int8_t*[output_channel_count];

  //The numberof output channel groups needed to compute the whole conv.
  //This is rounded up.
  int output_channel_groups = 
    (output_channel_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

  int dst_offset = 0;

  for(int ocg = 0; ocg < output_channel_groups; ++ocg){
    int output_channels_per_ocg = std::min(output_channel_count - ocg*vpu_ring_buffer_length, vpu_ring_buffer_length);

    int input_channel_groups = (bytes_per_output_channel + vpu_bytes_per_word - 1) / vpu_bytes_per_word;

    for(int icg = 0; icg < input_channel_groups; ++icg){

      int ocg_offset = ocg*vpu_ring_buffer_length;

      for (int out_ch = 0; out_ch < output_channels_per_ocg; ++out_ch){
        int bytes_in_this_vpu_copy = std::min(bytes_per_output_channel - icg*vpu_bytes_per_word, vpu_bytes_per_word);

        //reverse order of output channels
        int reversed_out_ch = output_channels_per_ocg - 1 - out_ch;
        int8_t * src = deref2d(raw_weights, bytes_per_output_channel, ocg_offset + reversed_out_ch, vpu_bytes_per_word*icg);
        int8_t * dst = boggled_weights + dst_offset;

        memcpy(dst, src, bytes_in_this_vpu_copy);
        dst_offset += bytes_in_this_vpu_copy;
        reordered_weights.weights.insert(reordered_weights.weights.end(), src, src + bytes_in_this_vpu_copy);
        
        if(icg == input_channel_groups-1){
          final_load_locations[ocg_offset + reversed_out_ch] = dst;
          reordered_weights.final_vpu_load_addresses[ocg_offset + reversed_out_ch] = dst;
        }
        
      }
    }
  }
  assert(dst_offset <= kernel_size);

  memset(boggled_weights + dst_offset, pad_value, kernel_size - dst_offset);
  // reordered_weights.weights.insert(reordered_weights.weights.end(), src, src + bytes_in_this_vpu_copy);

  return std::make_tuple(boggled_weights, final_load_locations, kernel_size);
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
  
  int32_t cur_output_channels_in_scope = std::min(output_slice_channel_count - output_channel_group * vpu_epv, vpu_epv);

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

  // std::cout << "k_p_adjust: " << k_p_adjust << std::endl;
  // std::cout << "input_channel_group_count: " << input_channel_group_count << std::endl;

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

  //Note: This forces kernels to be padded to word aligned boundaries TODO put an assert on this
  for(auto l=0; l< tail_loops; l++){
    VLMACCR(vpu, K_p);
    K_p += k_p_adjust;
  }

  VSTR(vpu, &A->vD);
  VSTD(vpu, &A->vR);
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

void MatMulFn::aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group)
{
#ifdef NN_USE_REF
  mat_mul_impl(A, T, output_channel_group);
#else
  mat_mul_impl_asm(A, T, output_channel_group);
#endif // NN_USE_REF
}
