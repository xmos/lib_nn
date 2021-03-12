#pragma once

#include "nn_types.h"
#include "util.hpp"
#include "../util/AddressCovector.hpp"

namespace nn {
namespace filt2d {
namespace geom {

template <typename T_elm = int8_t>
class ImageGeometry {

  public:

    unsigned const height;
    unsigned const width;
    unsigned const depth;

    ImageGeometry(
      unsigned const rows,
      unsigned const cols,
      unsigned const chans)
        : height(rows), width(cols), depth(chans){}

    unsigned const pixelElements() const { return this->depth; }
    unsigned const rowElements() const { return this->width * this->pixelElements(); }
    unsigned const imageElements() const { return this->height * this->rowElements(); }

    unsigned const pixelBytes() const { return sizeof(T_elm) * this->depth; }
    unsigned const rowBytes()   const { return pixelBytes()  * this->width;    }
    unsigned const imageBytes() const { return rowBytes()    * this->height;   }

    
    mem_stride_t getStride(ImageVect const& vect) const 
      { return this->getStride(vect.row, vect.col, vect.channel); }

    mem_stride_t getStride(int const rows, int const cols, int const chans) const 
      { return rows * this->rowBytes() + cols * this->pixelBytes() + chans * sizeof(T_elm); }

    AddressCovector<T_elm> getAddressCovector() const 
      { return AddressCovector<T_elm>(rowBytes(),pixelBytes(),sizeof(T_elm)); }
};




}}}