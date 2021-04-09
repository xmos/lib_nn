#pragma once

#include "nn_types.h"
#include "util.hpp"
#include "../util/AddressCovector.hpp"
#include "WindowGeometry.hpp"

#include <ostream>

namespace nn {
namespace filt2d {
namespace geom {

class ImageGeometry {

  public:

    using T_elm = int8_t;

    unsigned height;
    unsigned width;
    unsigned depth;

    constexpr ImageGeometry() : height(0), width(0), depth(0) {}

    constexpr ImageGeometry(
      unsigned const rows,
      unsigned const cols,
      unsigned const chans) noexcept
        : height(rows), width(cols), depth(chans){}

    constexpr ImageGeometry(
      ImageGeometry &X,
       WindowGeometry &K) noexcept : height(0), width(0), depth(0){

        // height = CONV2D_OUTPUT_LENGTH(X.height, K.shape.height, K.dilation.row, K.stride.row);
        // width = CONV2D_OUTPUT_LENGTH(X.width, K.shape.width, K.dilation.col, K.stride.col);
        // depth = X.depth; //TODO eek?
        }

    unsigned const pixelElements() const { return this->depth;                         }
    unsigned const rowElements()   const { return this->width * this->pixelElements(); }
    unsigned const imageElements() const { return this->height * this->rowElements();  }

    unsigned const pixelBytes() const { return sizeof(T_elm) * this->depth;  }
    unsigned const rowBytes()   const { return pixelBytes()  * this->width;  }
    unsigned const imageBytes() const { return rowBytes()    * this->height; }

    // unsigned const bitsPerElement() const { return rowBytes()    * this->height; }

    
    mem_stride_t getStride(const int rows, 
                           const int cols, 
                           const int chans) const;

    mem_stride_t getStride(const ImageVect& vect) const;

    mem_stride_t getStride(const ImageVect& from, 
                           const ImageVect& to) const;

    AddressCovector<T_elm> getAddressCovector() const;


      

    bool operator==(ImageGeometry other) const;
};


inline std::ostream& operator<<(std::ostream &stream, const ImageGeometry &image){
  return stream << image.height << ", " << image.width << ", " << image.depth;
}


}}}