#pragma once

#include "FilterGeometryIter.hpp"

namespace nn {
namespace test {

using namespace nn::ff;

namespace unpadded {

/**
 *
 *
 */
static inline nn::ff::FilterGeometryIterator AllUnpadded(
    const nn::Filter2dGeometry max_geometry, const bool depthwise = false,
    const int channel_step = 4) {
  return FilterGeometryIterator(
      {new nn::ff::AllUnpadded(max_geometry, depthwise, channel_step)});
}

/**
 * Iterates over output image shapes and window shapes with:
 *  - Window start = (0,0)
 *  - Window spatial strides = window spatial shape
 *  - Dilation = (0,0)
 * The input image's geometry will be calculated based on the output image
 * geometry and the window geometry such that the entire input image is consumed
 * and no padding is used.
 *
 * Only for depthwise filters
 *
 */
static inline nn::ff::FilterGeometryIterator SimpleDepthwise(
    const std::array<int, 2> output_spatial_range,
    const std::array<int, 2> window_spatial_range,
    const std::array<int, 2> channel_count_range, const int output_step = 1,
    const int window_step = 1, const int channel_step = 4) {
  return FilterGeometryIterator(
      nn::Filter2dGeometry(
          {output_spatial_range[0], output_spatial_range[0],
           channel_count_range[0]},
          {1, 1, 4},
          nn::WindowGeometry(window_spatial_range[0], window_spatial_range[0],
                             1, 0, 0, 0, 0, 1, 1, 1)),
      {new OutputShape({output_spatial_range[0], output_spatial_range[0],
                        channel_count_range[0]},
                       {output_spatial_range[1], output_spatial_range[1],
                        channel_count_range[1]},
                       {output_step, output_step, channel_step}),
       new WindowShape({window_spatial_range[0], window_spatial_range[0]},
                       {window_spatial_range[1], window_spatial_range[1]},
                       {window_step, window_step}),
       new Apply(nn::ff::MakeUnpaddedDepthwise)});
}

/**
 * Iterates over output image shapes and window shapes with:
 *  - Window start = (0,0)
 *  - Window spatial strides = window spatial shape
 *  - Dilation = (0,0)
 * The input image's geometry will be calculated based on the output image
 * geometry and the window geometry such that the entire input image is consumed
 * and no padding is used.
 *
 * Only for dense filters
 *
 */
static inline nn::ff::FilterGeometryIterator SimpleDense(
    const std::array<int, 2> output_spatial_range,
    const std::array<int, 2> window_spatial_range,
    const std::array<int, 2> out_channel_count_range,
    const std::array<int, 2> in_channel_count_range, const int output_step = 1,
    const int window_step = 1, const int out_channel_step = 4,
    const int in_channel_step = 4) {
  return FilterGeometryIterator(
      nn::Filter2dGeometry(
          {output_spatial_range[0], output_spatial_range[0],
           out_channel_count_range[0]},
          {1, 1, 4},
          nn::WindowGeometry(window_spatial_range[0], window_spatial_range[0],
                             1, 0, 0, 0, 0, 1, 1, 1)),
      {new OutputShape({output_spatial_range[0], output_spatial_range[0],
                        out_channel_count_range[0]},
                       {output_spatial_range[1], output_spatial_range[1],
                        out_channel_count_range[1]},
                       {output_step, output_step, out_channel_step}),
       new WindowShape({window_spatial_range[0], window_spatial_range[0]},
                       {window_spatial_range[1], window_spatial_range[1]},
                       {window_step, window_step}),
       new InputDepth(in_channel_count_range[0], in_channel_count_range[1],
                      in_channel_step),
       new Apply(nn::ff::MakeUnpaddedDense)});
}

}  // namespace unpadded

namespace padded {

/**
 *
 */
static inline nn::ff::FilterGeometryIterator AllPadded(
    const nn::Filter2dGeometry max_geometry, const nn::padding_t max_padding,
    const bool depthwise = false, const int channel_step = 4) {
  return FilterGeometryIterator(
      {new nn::ff::AllPaddedBase(max_geometry, depthwise, channel_step),
       new nn::ff::MakePadded(max_padding, depthwise)});
}
}  // namespace padded
}  // namespace test
}  // namespace nn