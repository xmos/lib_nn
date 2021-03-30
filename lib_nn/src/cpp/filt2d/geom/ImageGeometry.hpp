#pragma once

#include "nn_types.h"
#include "util.hpp"
#include "../util/AddressCovector.hpp"

#include <ostream>

namespace nn {

class ImageGeometry {

  public:

    unsigned height;
    unsigned width;
    unsigned depth;
    unsigned channel_depth; //bytes

    constexpr ImageGeometry() : height(0), width(0), depth(0), channel_depth(1) {}

    constexpr ImageGeometry(
      unsigned const rows,
      unsigned const cols,
      unsigned const chans,
      unsigned const channel_depth_bytes = 1) noexcept
        : height(rows), 
          width(cols), 
          depth(chans), 
          channel_depth(channel_depth_bytes) {}

    unsigned inline const pixelElements() const { return this->depth;                         }
    unsigned inline const rowElements()   const { return this->width * this->pixelElements(); }
    unsigned inline const colElements()   const { return this->height * this->pixelElements();}
    unsigned inline const imageElements() const { return this->height * this->rowElements();  }

    unsigned inline const pixelBytes() const { return pixelElements() * channel_depth; }
    unsigned inline const rowBytes()   const { return rowElements()   * channel_depth; }
    unsigned inline const colBytes()   const { return colElements()   * channel_depth; }
    unsigned inline const imageBytes() const { return imageElements() * channel_depth; }

    
    mem_stride_t getStride(const int rows, 
                           const int cols, 
                           const int chans) const;

    mem_stride_t getStride(const ImageVect& vect) const;

    mem_stride_t getStride(const ImageVect& from, 
                           const ImageVect& to) const;

    template <typename T>
    AddressCovector<T> getAddressCovector() const {
      //TODO: Should assert here that channel_depth == sizeof(T) ?
      return AddressCovector<T>(rowBytes(), pixelBytes(), channel_depth);
    }

    bool operator==(ImageGeometry other) const;
};


inline std::ostream& operator<<(std::ostream &stream, const ImageGeometry &image){
  return stream << image.height << ", " << image.width << ", " 
                << image.depth << ", " << image.channel_depth;
}


}