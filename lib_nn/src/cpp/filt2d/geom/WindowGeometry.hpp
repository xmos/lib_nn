#pragma once

#include "nn_types.h"
#include "../misc.hpp"
#include "../util/AddressCovector.hpp"

#include <cstdint>

namespace nn {
namespace filt2d {
namespace geom {


class WindowGeometry {

  using T_elm_in = int8_t;

  public:

    struct {
      unsigned height;
      unsigned width;
      unsigned depth;
    } shape;

    struct {
      int row;
      int col;
    } start;

    struct {
      int row;    
      int col;    
      int channel;
    } stride;

    struct {
      int row;
      int col;
    } dilation;


  public:

    constexpr WindowGeometry(
      unsigned const height,
      unsigned const width,
      unsigned const depth,
      int const start_row = 0,
      int const start_col = 0,
      int const stride_rows = 1,
      int const stride_cols = 1,
      int const stride_chans = 0,
      int const dil_rows = 1,
      int const dil_cols = 1) noexcept
        : shape{height, width, depth}, 
          start{start_row, start_col}, 
          stride{stride_rows, stride_cols, stride_chans}, 
          dilation{dil_rows, dil_cols} {}

    AddressCovector<T_elm_in> getPatchAddressCovector() const;

    unsigned pixelBytes() const;
    unsigned rowBytes() const;
    unsigned windowBytes() const;
    
    unsigned pixelElements() const;
    unsigned rowElements() const;
    unsigned windowElements() const;

    unsigned windowPixels() const;

    
    bool operator==(WindowGeometry other) const;
};



inline std::ostream& operator<<(std::ostream &stream, const WindowGeometry &window){
  return stream << "shape{" << window.shape.height << ", " << window.shape.width << ", " << window.shape.depth << "}, "
         << "start{" << window.start.row << ", " << window.start.col << "}, "
         << "stride{" << window.stride.row << ", " << window.stride.col << ", " << window.stride.channel << "}, "
         << "dilation{" << window.dilation.row << ", " << window.dilation.col << "}";
}

}}}