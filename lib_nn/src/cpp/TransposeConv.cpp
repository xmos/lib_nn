// Copyright 2023 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include "TransposeConv.h"

static unsigned deref(const unsigned r, const unsigned len, const unsigned idx){
     return len*r + idx;
}
static unsigned deref4d(const unsigned dim0, const unsigned dim1, 
            const unsigned dim2, const unsigned dim3, 
            const unsigned idx0, const unsigned idx1, 
            const unsigned idx2, const unsigned idx3){
    return deref(deref(deref(deref(0, dim0, idx0), dim1, idx1), dim2, idx2), dim3, idx3);
}

std::vector<ConvParams>  transpose_conv_reorder_kernel_weights(
    int8_t *raw_weights, 
    std::array<int, 4> &shape, 
    int32_t stride_height, //vertical
    int32_t stride_width  //horizontal
) {

  int32_t kernel_output_channels = shape[0];
  int32_t kernel_height = shape[1];
  int32_t kernel_width  = shape[2];
  int32_t kernel_input_channels = shape[3];

  std::vector<ConvParams> results;

  for (int h_idx=0;h_idx < stride_height; ++h_idx){
    int sub_kernel_height = (kernel_height + stride_height - 1 - h_idx) / stride_height;
    for (int w_idx=0; w_idx < stride_width; ++w_idx){
      int sub_kernel_width = (kernel_width + stride_width - 1 - w_idx) / stride_width;

      std::vector<int8_t> sub_kernel(kernel_output_channels*sub_kernel_height*sub_kernel_width*kernel_input_channels);
      
      for(int h_i = 0; h_i < sub_kernel_height; ++h_i ){
        for(int w_i = 0; w_i < sub_kernel_width; ++w_i ){
          int i = h_idx + stride_height*h_i;
          int j = w_idx + stride_width*w_i;

          for(int32_t ic = 0;ic < kernel_input_channels; ++ic){
            for(int32_t oc = 0;oc < kernel_output_channels; ++oc){

              int sk_index = deref4d(kernel_output_channels, sub_kernel_height, sub_kernel_width, kernel_input_channels, 
                    oc, sub_kernel_height - 1 - h_i, sub_kernel_width - 1 - w_i, ic);
              int rw_index = deref4d(kernel_output_channels, kernel_height, kernel_width, kernel_input_channels, 
                    oc, i, j, ic);

              sub_kernel[sk_index] = raw_weights[rw_index];
            }
          }
        }
      }

      std::array<int, 4> sub_shape = {kernel_output_channels, sub_kernel_height, sub_kernel_width, kernel_input_channels};
      ConvParams c(sub_shape, sub_kernel, h_idx, w_idx);
      results.push_back(c);
    }
  }
  return results;
}
