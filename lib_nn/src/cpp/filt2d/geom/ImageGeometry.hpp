#pragma once

#include "nn_types.h"
#include "util.hpp"
#include "../util/AddressCovector.hpp"

#include <iostream>
#include <cassert>

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
          
    unsigned inline const imagePixels()   const { return this->height * this->width;          }

    unsigned inline const pixelElements() const { return this->depth;                         }
    unsigned inline const rowElements()   const { return this->width * this->pixelElements(); }
    unsigned inline const colElements()   const { return this->height * this->pixelElements();}
    unsigned inline const imageElements() const { return this->height * this->rowElements();  }

    unsigned inline const pixelBytes() const { return pixelElements() * channel_depth; }
    unsigned inline const rowBytes()   const { return rowElements()   * channel_depth; }
    unsigned inline const colBytes()   const { return colElements()   * channel_depth; }
    unsigned inline const imageBytes() const { return imageElements() * channel_depth; }

    /**
     * Get the memory stride (in bytes) required to move by the specified amount within this geometry.
     */
    mem_stride_t getStride(const int rows, 
                           const int cols, 
                           const int chans) const;

    /**
     * Get the memory stride (in bytes) required to move by the specified amount within this geometry.
     */
    mem_stride_t getStride(const ImageVect& vect) const;

    /**
     * Get the memory stride (in bytes) required to move between two locations within this geometry.
     */
    mem_stride_t getStride(const ImageVect& from, 
                           const ImageVect& to) const;

    /**
     * Check whether the specified coordinates refer to an element within this image geometry.
     */
    bool IsWithinImage(const ImageVect& coords) const;
    
    /**
     * Check whether the specified coordinates refer to an element within this image geometry.
     */
    bool IsWithinImage(const int row, const int col, const int channel) const;

    /**
     * Get a reference to the specified element from the provided image.
     * An assertion is made that the specified location is within the geometry's bounds.
     */
    template <typename T>
    T& Element(T* img_base,
               const int row,
               const int col,
               const int channel) const;

    /**
     * Get the value of the specified element from the provided image.
     * If the coordinates refer to a location outside the image geometry pad_value is returned.
     */
    template <typename T>
    T Get(const T* img_base, 
          const ImageVect coords, 
          const T pad_value = 0) const;

    /**
     * Get the value of the specified element from the provided image.
     * If the coordinates refer to a location outside the image geometry pad_value is returned.
     */
    template <typename T>
    T Get(const T* img_base, 
          const int row, 
          const int col, 
          const int channel, 
          const T pad_value = 0) const;

    /**
     * Get an AddressCovector representing the geometry of this image.
     */
    template <typename T>
    AddressCovector<T> getAddressCovector() const {
      //TODO: Should assert here that channel_depth == sizeof(T) ?
      return AddressCovector<T>(rowBytes(), pixelBytes(), channel_depth);
    }

    /**
     * Determine whether this ImageGeometry is equal to another.
     */
    bool operator==(ImageGeometry other) const;
};




template <typename T>
T& ImageGeometry::Element(T* img_base, 
                          const int row,
                          const int col,
                          const int channel) const
{
  assert(IsWithinImage(row,col,channel));
  int index = row * this->width * this->depth + col * this->depth + channel;
  return img_base[index];
}




template <typename T>
T ImageGeometry::Get(const T* img_base, 
                     const ImageVect coords,
                     const T pad_value) const
{
  return Get<T>(img_base, coords.row, coords.col, coords.channel, pad_value);
}

template <typename T>
T ImageGeometry::Get(const T* img_base, 
                     const int row, 
                     const int col, 
                     const int channel,
                     const T pad_value) const
{
  if(!IsWithinImage(row, col, channel))
    return pad_value;
  return Element<T>(const_cast<T*>(img_base), row, col, channel);
}




inline std::ostream& operator<<(std::ostream &stream, const ImageGeometry &image){
  return stream << image.height << ", " << image.width << ", " 
                << image.depth << ", " << image.channel_depth;
}


}