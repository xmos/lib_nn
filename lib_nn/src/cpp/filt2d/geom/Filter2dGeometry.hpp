#pragma once

#include "util.hpp"
#include "ImageGeometry.hpp"
#include "WindowGeometry.hpp"

#include <cstdlib>
#include <cassert>




namespace nn {
  
  class Filter2dGeometry {

    public:

      using T_input = int8_t;
      using T_output = int8_t;

      ImageGeometry input;
      ImageGeometry output;
      WindowGeometry window;

    public:

      constexpr Filter2dGeometry(
        ImageGeometry input_geom,
        ImageGeometry output_geom,
        WindowGeometry window_geom) noexcept
          : input(input_geom), output(output_geom), window(window_geom) {}


      const ImageRegion GetFullJob() const;

      bool operator==(Filter2dGeometry other) const;
      bool operator!=(Filter2dGeometry other) const;

      bool ModelIsDepthwise() const;

      /**
       * Get the (signed or unsigned) spatial padding required by the filter corresponding to the
      * first (top-left) or last (bottom-right) output pixel.
      * 
      * Unsigned padding is signed padding with std::max(0,_) applied to each of its values. With
      * unsigned padding, a value of 0 indicates padding is not required, and a positive value 
      * indicates the number of pixels of padding that are required along the border 
      * (top/left/bottom/right).
      * 
      * Signed padding is useful, for example, in calculations where the padding requirement of a 
      * particular output pixel is required. In this case, negative values indicate the distance to the
      * corresponding border of the input image. A signed padding value of 0 indicates the convolution 
      * window is already at the edge of the image, whereas -1 indicates there is one pixel between that
      * edge and the current convolution window.
      * 
      * Using `initial = true` will retrieve the padding for the top-left output pixel, and `false` 
      * for the bottom-right output pixel. The maximum value (per each padding direction) between the
      * initial and final output pixels is the maximum amount of padding needed (for that direction), as
      * all other output pixels are guaranteed to need equal to or less than that amount.
      * 
      * TODO: This doesn't currently handle dilation correctly!!
      */
      padding_t ModelPadding(bool initial = true, 
                            bool signed_padding = true) const;

      /**
       * Does every output pixel have the convolution window entirely within the bounds of 
      * the input image?
      */
      bool ModelRequiresPadding() const;

      /**
       * Is it true that, for any given output pixel, the convolution window is guaranteed to have
      * at least one pixel within the input image's bounds?
      */
      bool ModelConvWindowAlwaysIntersectsInput() const;

      /**
       * Does the filter's geometry imply that every input element is used to compute some
      * output element?
      */
      bool ModelConsumesInput() const;

  };



  inline std::ostream& operator<<(std::ostream &stream, const Filter2dGeometry &filt){
    return stream << "input{" << filt.input << "}, output{" << filt.output << "}, window{" << filt.window << "}";
  }

}