#pragma once

#include <cmath>
#include <limits>

#include "Filter2D.hpp"

namespace nn {

class MaxPool2d : public ChannelParallelComponent<VPU_INT8_EPV_LOG2> {
 public:
  /**
   * Function can be used with WindowLocation::Fold() to apply the MaxPool2d for
   * a particular output.
   */
  template <typename T>
  static T FoldFunc(const ImageVect& filter_coords,
                    const ImageVect& input_coords, const T prev_max,
                    const T new_elm, const bool is_padding) {
    return std::max<T>(prev_max, new_elm);
  }
};

/**
 * MaxPool2d operator that should work for (almost) any geometry.
 *
 * Where supported, MaxPool2d_Valid should generally be used. In particular,
 *this operator supports geometries that involve input padding or dilations
 *other than 1.
 *
 * @see MaxPool2d_Valid
 **/
class MaxPool2d_Generic : public Filter2D_DW, public MaxPool2d {
 public:
  struct Params {
    AbstractKernel::Params ak_params;
    ImToColPadded::Params mem_params;
    MaxPoolPatchFn::Params agg_params;
    DirectWriteOutputTransform::Params ot_params;

    Params(const Filter2dGeometry& filter_geometry,
           const ImageRegion& output_region,
           const int8_t padding_value = std::numeric_limits<int8_t>::min());
  };

 public:
  static bool SupportsGeometry(const Filter2dGeometry& filter_geometry);

 public:
  MaxPool2d_Generic() = delete;
  MaxPool2d_Generic(MaxPool2d_Generic&) = delete;
  MaxPool2d_Generic(MaxPool2d_Generic&&) = delete;

  MaxPool2d_Generic(ImToColPadded* memcopy,
                    MaxPoolPatchFn* agg, DirectWriteOutputTransform* ot);
};

/**
 * MaxPool2d operator that works for geometries which involve no input padding
 *and where both vertical and horizontal dilation are 1.
 **/
class MaxPool2d_Valid : public Filter2D_DW, public MaxPool2d {
 public:
  struct Params {
    AbstractKernel::Params ak_params;
    DerefInputFn::Params mem_params;
    MaxPoolDirectValidFn::Params agg_params;
    DirectWriteOutputTransform::Params ot_params;

    Params(const Filter2dGeometry& filter_geometry,
           const ImageRegion& output_region);
  };

 public:
  static bool SupportsGeometry(const Filter2dGeometry& filter_geometry);

 public:
  MaxPool2d_Valid() = delete;
  MaxPool2d_Valid(MaxPool2d_Valid&) = delete;
  MaxPool2d_Valid(MaxPool2d_Valid&&) = delete;

  MaxPool2d_Valid(DerefInputFn* memcopy_handler,
                  MaxPoolDirectValidFn* aggregate_handler,
                  DirectWriteOutputTransform* ot_handler);
};

}  // namespace nn
