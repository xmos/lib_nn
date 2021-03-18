#pragma once

#include "nn_types.h"
#include "util.hpp"
#include "../util/AddressCovector.hpp"

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

    constexpr ImageGeometry(
      unsigned const rows,
      unsigned const cols,
      unsigned const chans) noexcept
        : height(rows), width(cols), depth(chans){}

    unsigned const pixelElements() const { return this->depth;                         }
    unsigned const rowElements()   const { return this->width * this->pixelElements(); }
    unsigned const imageElements() const { return this->height * this->rowElements();  }

    unsigned const pixelBytes() const { return sizeof(T_elm) * this->depth;  }
    unsigned const rowBytes()   const { return pixelBytes()  * this->width;  }
    unsigned const imageBytes() const { return rowBytes()    * this->height; }

    
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