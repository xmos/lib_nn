#pragma once

#include "Filter2D.hpp"

#include <cmath>
#include <limits>

namespace nn {

  class AvgPool2d : public ChannelParallelComponent<VPU_INT8_ACC_PERIOD_LOG2> { 
    public:

      static void ComputeScaleShift(const WindowGeometry& window,
                                    int8_t& input_scale,
                                    int16_t& output_shift);
      static void ComputeScaleShift(const int window_pixels,
                                    int8_t& input_scale,
                                    int16_t& output_shift);
  };

  /**
   * AvgPool2d operator that should work for (almost) any geometry.
   * 
   * Where supported, AvgPool2d_Valid should generally be used. In particular, this operator
   * supports geometries that involve input padding or dilations other than 1.
   * 
   * @see AvgPool2d_Valid
   **/
  class AvgPool2d_Generic : public Filter2D_DW, public AvgPool2d {

    public:

      struct Params {
        AbstractKernel::Params ak_params;
        ImToColPadded::Params  mem_params;
        AvgPoolPatchFn::Params agg_params;
        ShiftInt8OutputTransform::Params ot_params;

        Params(const Filter2dGeometry& filter_geometry,
               const ImageRegion& output_region,
               const int8_t padding_value = std::numeric_limits<int8_t>::min() );
      };

    public:

      static bool SupportsGeometry(const Filter2dGeometry& filter_geometry);

    public: 

      AvgPool2d_Generic() = delete;
      AvgPool2d_Generic(AvgPool2d_Generic&) = delete;
      AvgPool2d_Generic(AvgPool2d_Generic&&) = delete;

      AvgPool2d_Generic(AbstractKernel::Params* params,
                        ImToColPadded* memcopy,
                        AvgPoolPatchFn* agg,
                        ShiftInt8OutputTransform* ot,
                        int8_t* scratch_mem);
  };

  /**
   * AvgPool2d operator that works for geometries which involve no input padding and 
   * where both vertical and horizontal dilation are 1.
   **/
  class AvgPool2d_Valid : public Filter2D_DW, public AvgPool2d {

    public:

      struct Params {
        AbstractKernel::Params ak_params;
        DerefInputFn::Params  mem_params;
        AvgPoolDirectValidFn::Params agg_params;
        ShiftInt8OutputTransform::Params ot_params;

        Params(const Filter2dGeometry& filter_geometry,
               const ImageRegion& output_region);
      };

    public:

      static bool SupportsGeometry(const Filter2dGeometry& filter_geometry);

    public: 

      AvgPool2d_Valid() = delete;
      AvgPool2d_Valid(AvgPool2d_Valid&) = delete;
      AvgPool2d_Valid(AvgPool2d_Valid&&) = delete;

      AvgPool2d_Valid(AbstractKernel::Params* params,
                      DerefInputFn* memcopy_handler,
                      AvgPoolDirectValidFn* aggregate_handler,
                      ShiftInt8OutputTransform* ot_handler);
  };


}
