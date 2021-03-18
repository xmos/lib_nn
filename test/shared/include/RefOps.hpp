#pragma once

#include <vector>
#include <cstdint>

#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"

namespace nn {
  namespace test {
    namespace ops {
      namespace ref {

struct Conv2dDenseParams {
  struct { 
    int32_t zero_point; 
  } input;
  struct {
    int32_t zero_point;
    float* effective_multiplier;
  } output;
};

std::vector<int8_t> Conv2dDenseReference(
    const nn::filt2d::geom::Filter2dGeometry& filter_geometry,
    const int8_t input_img[],
    const int8_t kernel_weights[],
    const int32_t biases[],
    const float effective_output_multiplier[],
    const int8_t input_zero_point,
    const int8_t output_zero_point);

}}}}