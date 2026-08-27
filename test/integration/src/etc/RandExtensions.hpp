// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <cmath>
#include <cstdint>
#include <limits>

#include "Rand.hpp"
#include "geom/Filter2dGeometry.hpp"

namespace nn {
namespace test {

struct RandFilterParams {
  bool allow_padding;
  bool allow_dilation;
  bool is_depthwise;
  unsigned depthwise_channels_per_output_group;

  struct {
    struct {
      unsigned min, max, step;
    } depth;
  } input, output;
};

}  // namespace test
}  // namespace nn

template <>
inline nn::Filter2dGeometry
nn::test::Rand::rand<nn::Filter2dGeometry, nn::test::RandFilterParams>(
    nn::test::RandFilterParams params) {
  // Some of the stuff below should be parameterized in the future

  unsigned K_h = this->rand<unsigned>(2, 5);
  unsigned K_w = this->rand<unsigned>(2, 5);

  unsigned Y_h = this->rand<unsigned>(1, 10);
  unsigned Y_w = this->rand<unsigned>(1, 10);

  unsigned X_c = params.input.depth.min +
                 params.input.depth.step *
                     this->rand<unsigned>(
                         0, (params.input.depth.max - params.input.depth.min) /
                                params.input.depth.step);
  unsigned Y_c = params.output.depth.min +
                 params.output.depth.step *
                     this->rand<unsigned>(0, (params.output.depth.max -
                                              params.output.depth.min) /
                                                 params.output.depth.step);

  // Except, if this is depthwise, Y_c must be X_c
  if (params.is_depthwise) Y_c = X_c;

  // Need to make sure at least 1 window element is within the input image,
  //  or all of them if padding isn't allowed
  int W_start_row_min = params.allow_padding ? 1 - int(K_h) : 0;
  int W_start_col_min = params.allow_padding ? 1 - int(K_w) : 0;

  int W_start_row = this->rand<int>(W_start_row_min, 0);
  int W_start_col = this->rand<int>(W_start_col_min, 0);

  int W_stride_row = this->rand<int>(1, K_h);
  int W_stride_col = this->rand<int>(1, K_w);

  int W_dilation_row = params.allow_dilation ? this->rand<int>(1, 3) : 1;
  int W_dilation_col = params.allow_dilation ? this->rand<int>(1, 3) : 1;

  // The input image should be small enough that we use all of it to calculate
  // the output
  //  and large enough that the last output row/col still has some intersection
  //  with the input window

  // So, the maximum X dimensions are those that place the last row/col of the
  // conv window in the last row/col
  //  of X for the last output row/col
  unsigned X_h_max =
      W_start_row + (Y_h - 1) * W_stride_row + (K_h - 1) * W_dilation_row;
  unsigned X_w_max =
      W_start_col + (Y_w - 1) * W_stride_col + (K_w - 1) * W_dilation_col;

  // The min X dimensions are those that place the first row/col of the conv
  // window in the last row/col of
  //  X for the last output row/col.
  unsigned X_h_min =
      (unsigned)std::min(W_start_row + (int(Y_h) - 1) * W_stride_row, 1);
  unsigned X_w_min =
      (unsigned)std::min(W_start_col + (int(Y_w) - 1) * W_stride_col, 1);

  // Except if padding isn't allowed, the min/max X are actually the same as the
  // min
  unsigned X_h =
      params.allow_padding ? this->rand<unsigned>(X_h_min, X_h_max) : X_h_min;
  unsigned X_w =
      params.allow_padding ? this->rand<unsigned>(X_w_min, X_w_max) : X_w_min;

  unsigned K_c =
      params.is_depthwise ? params.depthwise_channels_per_output_group : X_c;

  return nn::Filter2dGeometry(
      nn::ImageGeometry(X_h, X_w, X_c), nn::ImageGeometry(Y_h, Y_w, Y_c),
      nn::WindowGeometry(K_h, K_w, K_c, W_start_row, W_start_col, W_stride_row,
                         W_stride_col, W_dilation_row, W_dilation_col));
}

namespace nn {
namespace test {

struct SimpleFilter {
  bool depthwise;
  bool padded;
  unsigned cog_size;

  SimpleFilter(bool depthwise = false, bool padded = false,
               unsigned cog_size = 16)
      : depthwise(depthwise), padded(padded), cog_size(cog_size) {}
};

}  // namespace test
}  // namespace nn

template <>
inline nn::Filter2dGeometry
nn::test::Rand::rand<nn::Filter2dGeometry, nn::test::SimpleFilter>(
    nn::test::SimpleFilter p) {
  if (p.padded) {
    const int K_h = this->rand<int>(2, 4);
    const int K_w = this->rand<int>(2, 4);

    const int pad_t = this->rand<int>(1, K_h - 1);
    const int pad_l = this->rand<int>(1, K_w - 1);
    const int pad_b = this->rand<int>(1, K_h - 1);
    const int pad_r = this->rand<int>(1, K_w - 1);

    const int Y_h = this->rand<int>(2, 10);
    const int Y_w = this->rand<int>(2, 10);

    const int X_h = Y_h * K_h - (pad_t + pad_b);
    const int X_w = Y_w * K_w - (pad_l + pad_r);

    const int X_c = this->rand<int>(1, 10) * 4;
    const int Y_c = p.depthwise ? X_c : (this->rand<int>(1, 10) * 4);
    const int W_c = p.depthwise ? 1 : X_c;
    const int W_stride_chan = p.depthwise ? 1 : 0;

    return nn::Filter2dGeometry(
        nn::ImageGeometry(X_h, X_w, X_c), nn::ImageGeometry(Y_h, Y_w, Y_c),
        nn::WindowGeometry(K_h, K_w, W_c, -pad_t, -pad_l, K_h, K_w,
                           W_stride_chan));

  } else {
    const int K_h = this->rand<int>(3, 3);
    const int K_w = this->rand<int>(4, 4);

    const int Y_h = this->rand<int>(1, 10);
    const int Y_w = this->rand<int>(1, 10);

    const int X_h = Y_h * K_h;
    const int X_w = Y_w * K_w;

    const int X_c = this->rand<int>(1, 10) * 4;
    const int Y_c = p.depthwise ? X_c : (this->rand<int>(1, 10) * 4);
    const int W_c = p.depthwise ? 1 : X_c;
    const int W_stride_chan = p.depthwise ? 1 : 0;

    return nn::Filter2dGeometry(
        nn::ImageGeometry(X_h, X_w, X_c), nn::ImageGeometry(Y_h, Y_w, Y_c),
        nn::WindowGeometry(K_h, K_w, W_c, 0, 0, K_h, K_w, W_stride_chan));
  }
}

// Rounds and saturates to int8_t, matching the reference kernels' rounding.
static inline int8_t round_int8(float val) {
  auto r = std::roundf(val + ldexpf(((val >= 0) ? 1 : -1), -10));
  return int8_t(
      std::max<float>(std::numeric_limits<int8_t>::min(),
                      std::min<float>(std::numeric_limits<int8_t>::max(), r)));
}