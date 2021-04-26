#pragma once

#include "nn_types.h"
#include "geom/util.hpp"
#include "ImageGeometry.hpp"

#include <cstdint>

namespace nn
{

  class WindowGeometry
  {

  public:
    ImageGeometry shape;

    struct
    {
      int row;
      int col;
    } start;

    struct
    {
      int row;
      int col;
      int channel;
    } stride;

    struct
    {
      int row;
      int col;
    } dilation;

  public:
    constexpr WindowGeometry(
        int const height,
        int const width,
        int const depth,
        int const start_row = 0,
        int const start_col = 0,
        int const stride_rows = 1,
        int const stride_cols = 1,
        int const stride_chans = 0,
        int const dil_rows = 1,
        int const dil_cols = 1,
        int const channel_depth = 1) noexcept
        : shape(height, width, depth, channel_depth), //asj: channel_depth could be bits_per_element
          start{start_row, start_col},
          stride{stride_rows, stride_cols, stride_chans},
          dilation{dil_rows, dil_cols}
    {
    }

    bool operator==(WindowGeometry other) const;

    /**
       * Get the coordinates (in an input image's coordinate space) of the first (i.e. top-left) pixel of the
       * filter window corresponding to the specified output element.
       */
    ImageVect WindowCoords(const ImageVect &output_coords) const;

    /**
       * Does the filter use a dilation other than 1?
       */
    inline bool UsesDilation() const { return (dilation.row != 1) || (dilation.col != 1); }
  };

  inline std::ostream &operator<<(std::ostream &stream, const WindowGeometry &window)
  {
    return stream << "shape{" << window.shape.height << "," << window.shape.width << "," << window.shape.depth << "},"
                  << "start{" << window.start.row << "," << window.start.col << "},"
                  << "stride{" << window.stride.row << "," << window.stride.col << "," << window.stride.channel << "},"
                  << "dilation{" << window.dilation.row << "," << window.dilation.col << "}";
  }

}