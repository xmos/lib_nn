#pragma once

#include "util.hpp"
#include "ImageGeometry.hpp"
#include "WindowGeometry.hpp"

#include <cstdlib>
#include <cassert>

namespace nn
{

  class WindowLocation;

  class Filter2dGeometry
  {

  public:
    ImageGeometry input;
    ImageGeometry output;
    WindowGeometry window;

  public:
    constexpr Filter2dGeometry() noexcept : input(), output(), window() {}

    constexpr Filter2dGeometry(
        ImageGeometry input_geom,
        ImageGeometry output_geom,
        WindowGeometry window_geom) noexcept
        : input(input_geom),
          output(output_geom),
          window(window_geom) {}

    const ImageRegion GetFullJob() const;

    bool operator==(Filter2dGeometry other) const;
    bool operator!=(Filter2dGeometry other) const;

    WindowLocation GetWindow(const ImageVect output_coords) const;
    WindowLocation GetWindow(const int row, const int col, const int channel) const;

    bool ModelIsDepthwise() const;

    /**
       * Get the total implied padding around each edge of the input image.
       * 
       * Note: This is the unsigned padding
       */
    padding_t Padding() const;

    //TODO
    int getReceptiveVolumeElements() const { return window.shape.height * window.shape.width * input.pixelElements(); }

    int getReceptiveVolumeBytes() const { return window.shape.height * window.shape.width * input.pixelBytes(); }

    /**
       * @Deprecated
       * @TODO: Get rid of this when possible. There are a couple test cases that still use it.
      */
    padding_t ModelPadding(bool initial = true,
                           bool signed_padding = true) const;
  };

  inline std::ostream &operator<<(std::ostream &stream, const Filter2dGeometry &filt)
  {
    return stream << "input{" << filt.input << "}, output{" << filt.output << "}, window{" << filt.window << "}";
  }

}

#include "WindowLocation.hpp"
