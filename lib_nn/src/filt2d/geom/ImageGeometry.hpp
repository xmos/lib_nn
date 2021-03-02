#pragma once

#include "../Filter2d_util.hpp"

namespace nn {
namespace filt2d {
namespace geom {

template <typename T_elm = int8_t>
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
    unsigned const rowBytes()   const { return pixelBytes()  * this->width;    }
    unsigned const imageBytes() const { return rowBytes()    * this->height;   }

    
    mem_stride_t getStride(ImageVect const& vect) const 
      { return this->getStride(vect.row, vect.col, vect.channel); }

    mem_stride_t getStride(int const rows, int const cols, int const chans) const 
      { return rows * this->rowBytes() + cols * this->pixelBytes() + chans * sizeof(T_elm); }

    PointerCovector getPointerCovector() const 
      { return PointerCovector(rowBytes(),pixelBytes(),sizeof(T_elm)); }
};




}}}