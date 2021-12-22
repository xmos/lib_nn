#pragma once

#include <cassert>
#include <cstdint>
#include <functional>
#include <iostream>

#include "nn_types.h"
#include "util.hpp"

namespace nn {

// TODO: [astew] CHAR_BIT not defined if I build with Cygwin or gcc+Ubuntu+WSL.
// Not in <limits> either.
#ifndef CHAR_BIT
#define CHAR_BIT (sizeof(char) * 8)
#endif

/**
 * Represents the geometry of a 2D multi-channel image.
 */
class ImageGeometry {
 public:
  /// Height of image in rows
  int height;
  /// Width of image in columns
  int width;
  /// Depth of image in channels
  int depth;
  /// Number of bits per image element
  int element_bits;

  /**
   * Default Constructor
   *
   * Dimensions are initialized to 0, with 1 byte per pixel.
   */
  constexpr ImageGeometry()
      : height(0), width(0), depth(0), element_bits(CHAR_BIT) {}

  /**
   * Construct an ImageGeometry with the specified dimensions. Defaults to 8
   * bits per element.
   */
  constexpr ImageGeometry(int const rows, int const cols, int const chans,
                          int const element_bits = CHAR_BIT) noexcept
      : height(rows), width(cols), depth(chans), element_bits(element_bits) {}

  /**
   * The total number of pixels in the image, i.e. height x width
   */
  int inline const PixelCount() const { return this->height * this->width; }

  /**
   * The number of image elements per pixel, i.e. channels
   */
  int inline const PixelElements() const { return this->depth; }

  /**
   * The number of image elements per row of the image, i.e. width x channels
   */
  int inline const RowElements() const {
    return this->width * this->PixelElements();
  }

  /**
   * The number of image elements per column of the image, i.e. height x
   * channels
   */
  int inline const ColElements() const {
    return this->height * this->PixelElements();
  }

  /**
   * The total number of image elements, i.e. width x height x channels
   */
  int inline const ElementCount() const {
    return this->height * this->RowElements();
  }

  /**
   * The number of image pixels multiplied by the square of image depth
   */
  int inline const VolumeElements() const {
    return this->depth * this->ElementCount();
  }

  /**
   * The number of bytes per image pixel
   */
  int inline const PixelBytes() const {
    return PixelElements() * element_bits / CHAR_BIT;
  }

  /**
   * The number of bytes per row of the image
   */
  int inline const RowBytes() const {
    return RowElements() * element_bits / CHAR_BIT;
  }

  /**
   * The number of bytes per column of the image
   */
  int inline const ColBytes() const {
    return ColElements() * element_bits / CHAR_BIT;
  }

  /**
   * The total number of bytes of the image
   */
  int inline const ImageBytes() const {
    return ElementCount() * element_bits / CHAR_BIT;
  }

  /**
   * Get the flattened index of the specified image element.
   *
   * The "flattened" index of an element is the index of the element when the
   * image is stored in a 1 dimensional array. This is ideal, for example, when
   * the image is backed by a `std::vector` object.
   *
   * This function returns -1 if the specified coordinates refer to an element
   * in padding (i.e. beyond the bounds of the image).
   */
  int Index(const int row, const int col, const int channel) const;

  /**
   * Get the buffer index associated with the specified image coordinates,
   * assuming the image is backed by a linear buffer.
   */
  int Index(const ImageVect &input_coords) const;

  /**
   * Get the memory stride (in bytes) required to move by the specified amount
   * within this geometry.
   */
  mem_stride_t GetStride(const int rows, const int cols, const int chans) const;

  /**
   * Get the memory stride (in bytes) required to move by the specified amount
   * within this geometry.
   */
  mem_stride_t GetStride(const ImageVect &vect) const;

  /**
   * Get the memory stride (in bytes) required to move between two locations
   * within this geometry.
   */
  mem_stride_t GetStride(const ImageVect &from, const ImageVect &to) const;

  /**
   * Check whether the specified coordinates refer to an element within this
   * image geometry.
   */
  bool IsWithinImage(const ImageVect &coords) const;

  /**
   * Check whether the specified coordinates refer to an element within this
   * image geometry.
   */
  bool IsWithinImage(const int row, const int col, const int channel) const;

  /**
   * Get a reference to the specified element from the provided image.
   * An assertion is made that the specified location is within the geometry's
   * bounds.
   */
  template <typename T>
  T &Element(T *img_base, const int row, const int col,
             const int channel) const;

  /**
   * Get the value of the specified element from the provided image.
   * If the coordinates refer to a location outside the image geometry pad_value
   * is returned.
   */
  template <typename T>
  T Get(const T *img_base, const ImageVect &coords,
        const T pad_value = 0) const;

  /**
   * Get the value of the specified element from the provided image.
   * If the coordinates refer to a location outside the image geometry pad_value
   * is returned.
   */
  template <typename T>
  T Get(const T *img_base, const int row, const int col, const int channel,
        const T pad_value = 0) const;

  /**
   * Apply an operation per element of an image.
   *
   * void PerPixelOp(const int row,
   *                 const int col,
   *                 const int channel,
   *                 T& value);
   */
  template <typename T>
  void ApplyOperation(T *image_base,
                      std::function<void(int, int, int, T &)> func) const;

  /**
   * Determine whether this ImageGeometry is equal to another.
   */
  bool operator==(ImageGeometry other) const;
};

/////////////////////////
template <typename T>
T &ImageGeometry::Element(T *img_base, const int row, const int col,
                          const int channel) const {
  assert(IsWithinImage(row, col, channel));
  int index = row * this->width * this->depth + col * this->depth + channel;
  return img_base[index];
}

/////////////////////////
template <typename T>
T ImageGeometry::Get(const T *img_base, const ImageVect &coords,
                     const T pad_value) const {
  return Get<T>(img_base, coords.row, coords.col, coords.channel, pad_value);
}

/////////////////////////
template <typename T>
T ImageGeometry::Get(const T *img_base, const int row, const int col,
                     const int channel, const T pad_value) const {
  if (!IsWithinImage(row, col, channel)) return pad_value;
  return Element<T>(const_cast<T *>(img_base), row, col, channel);
}

/////////////////////////
template <typename T>
void ImageGeometry::ApplyOperation(
    T *image_base, std::function<void(int, int, int, T &)> op) const {
  for (int row = 0; row < this->height; ++row)
    for (int col = 0; col < this->width; ++col)
      for (int channel = 0; channel < this->depth; ++channel)
        op(row, col, channel, this->Element<T>(image_base, row, col, channel));
}

/////////////////////////
inline std::ostream &operator<<(std::ostream &stream,
                                const ImageGeometry &image) {
  return stream << image.height << "," << image.width << "," << image.depth
                << "," << image.element_bits / CHAR_BIT;
}

}  // namespace nn
