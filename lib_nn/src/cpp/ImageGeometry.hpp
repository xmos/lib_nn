#pragma once

#include "f2d_c_types.h"
#include "Filter2D_util.hpp"

namespace nn {

template <typename T_elm>
class ImageGeometry {

  public:

    unsigned const height;
    unsigned const width;
    unsigned const channels;

    ImageGeometry(
      unsigned const rows,
      unsigned const cols,
      unsigned const chans)
        : height(rows), width(cols), channels(chans){}


    unsigned const pixelBytes() const { return sizeof(T_elm) * this->channels; }
    unsigned const rowBytes() const { return pixelBytes() * this->width; }
    unsigned const imageBytes() const { return rowBytes() * this->height; }

    
    mem_stride_t getStride(
      ImageVect<int32_t> const& vect)
    {
      return vect.row * this->rowBytes() 
           + vect.col * this->pixelBytes() 
           + vect.channel * sizeof(T_elm);
    }
};


}