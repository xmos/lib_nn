// Copyright 2021 XMOS LIMITED. This Software is subject to the terms of the
// XMOS Public License: Version 1
#include <array>
#include <cstdint>
#include <vector>

class ConvParams {    
  public: 
    ConvParams(std::array<int, 4> ks, std::vector<int8_t> weights, int h, int w) : 
     subH(h), subW(w), kernelShape(ks) , weights(weights){}

  public:     
    int subH;
    int subW;
    std::array<int, 4> kernelShape;
    std::vector<int8_t> weights;
};

std::vector<ConvParams> transpose_conv_reorder_kernel_weights(
    int8_t *raw_weights, 
    std::array<int, 4> &shape, 
    int32_t stride_height, //vertical
    int32_t stride_width  //horizontal
) ;
