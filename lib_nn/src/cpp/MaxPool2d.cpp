
#include "MaxPool2d.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>

using namespace nn;

MaxPool2d_Generic::Params::Params(const Filter2dGeometry& filter,
                                  const ImageRegion& region,
                                  const int8_t padding_value)
    : ak_params(filter.output, region,
                MaxPool2d_Generic::ChannelsPerOutputGroup),
      mem_params(filter, padding_value,
                 MaxPool2d_Generic::ChannelsPerOutputGroup),
      agg_params(filter.window),
      ot_params(filter.output) {}

MaxPool2d_Valid::Params::Params(const Filter2dGeometry& filter,
                                const ImageRegion& region)
    : ak_params(filter.output, region,
                MaxPool2d_Generic::ChannelsPerOutputGroup),
      mem_params(filter),
      agg_params(filter.input, filter.window),
      ot_params(filter.output) {}

bool MaxPool2d_Generic::SupportsGeometry(const Filter2dGeometry& filter) {
  const auto& input = filter.input;
  const auto& output = filter.output;
  const auto& window = filter.window;

  /// TODO: Some basic sanity checks should probably be collected into a
  /// function elsewhere, like
  ///       making sure image shapes are positive.

  if (input.height <= 0 || input.width <= 0 || input.depth <= 0 ||
      input.channel_depth <= 0)
    return false;
  if (output.height <= 0 || output.width <= 0 || output.depth <= 0 ||
      output.channel_depth <= 0)
    return false;
  if (window.shape.height <= 0 || window.shape.width <= 0 ||
      window.shape.depth <= 0 || window.shape.channel_depth <= 0)
    return false;

  // Input and output images must have the same number of channels.
  if (input.depth != output.depth) return false;

  // And the elements must be the same size
  if (input.channel_depth != output.channel_depth) return false;

  // Window depth and channel stride must each be exactly 1
  if (window.shape.depth != 1 || window.stride.channel != 1) return false;

  // Window channel depth must equal input chanel depth.
  if (window.shape.channel_depth != input.channel_depth) return false;

  // Channel count must be a multiple of 4 to guarantee correct alignment
  if (input.depth % 4 != 0) return false;

  // There must be at least one pixel of the filter window intersecting
  // with the input image for every output pixel location.

  auto loc1 = filter.GetWindow(0, 0, 0);
  auto cc =
      loc1.InputCoords(window.shape.height - 1, window.shape.width - 1, 0);
  if (cc.row < 0) return false;
  if (cc.col < 0) return false;
  auto loc2 = filter.GetWindow(output.height - 1, output.width - 1, 0);
  cc = loc2.InputCoords(0, 0, 0);
  if (cc.row >= input.height) return false;
  if (cc.col >= input.width) return false;

  // Otherwise, it's supported
  return true;
}

bool MaxPool2d_Valid::SupportsGeometry(const Filter2dGeometry& filter) {
  // Geometries supported by MaxPool2d_Valid are a strict subset of those
  // supported by MaxPool2d_Generic
  if (!MaxPool2d_Generic::SupportsGeometry(filter)) return false;

  // Padding is not supported
  if (filter.Padding().HasPadding()) return false;

  // // Dilation other than 1 isn't supported
  // if( window.dilation.row != 1 || window.dilation.col != 1 ) return false;

  // Otherwise, it's supported
  return true;
}

MaxPool2d_Generic::MaxPool2d_Generic(AbstractKernel::Params* ak_params,
                                     ImToColPadded* memcopy_handler,
                                     MaxPoolPatchFn* aggregate_handler,
                                     DirectWriteOutputTransform* ot_handler)

    : Filter2D_DW(ak_params, memcopy_handler, aggregate_handler, ot_handler,
                  ChannelsPerOutputGroup) {}

MaxPool2d_Valid::MaxPool2d_Valid(AbstractKernel::Params* ak_params,
                                 DerefInputFn* memcopy_handler,
                                 MaxPoolDirectValidFn* aggregate_handler,
                                 DirectWriteOutputTransform* ot_handler)
    : Filter2D_DW(ak_params, memcopy_handler, aggregate_handler, ot_handler,
                  ChannelsPerOutputGroup) {}