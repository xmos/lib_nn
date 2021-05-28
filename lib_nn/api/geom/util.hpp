#pragma once

#include <array>
#include <cstdint>
#include <iostream>

#include "nn_api.h"

#define CONV2D_OUTPUT_LENGTH(input_length, filter_size, dilation, stride)     \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)

namespace nn {

/**
 * Represents padding for a filter (Filter2dGeometry::Padding()) or receptive
 * field (WindowLocation::Padding()).
 *
 * The padding may be signed or unsigned. Signed padding may include negative
 * values, which effectively indicate the buffer before encountering the
 * corresponding edge of the input image (e.g.  `left == -4` for a particular
 * receptive field indicates that there are 4 pixels between the left edge of
 * the receptive field and the left edge of the input image).
 *
 * Unsigned padding is similar, but negative values are 0 instead.
 */
C_API
typedef struct padding_t {
  /// Rows of padding associated with the top edge (i.e. low row values)
  int16_t top;
  /// Columns of padding associated with the left edge  (i.e. low column values)
  int16_t left;
  /// Rows of padding associated with the bottom edge (i.e. higher row values)
  int16_t bottom;
  /// Columns of padding associated with the right edge  (i.e. higher column
  /// values)
  int16_t right;

  /**
   * Replace negative fields with 0.
   */
  void MakeUnsigned();

  /**
   * Indicates whether padding is required in any direction.
   */
  bool HasPadding() const;

  /**
   * Test for equality of two `padding_t`.
   *
   * Two `padding_t` are equal iff all of their (corresponding) fields are
   * equal.
   */
  bool operator==(const padding_t &other) const;

  /**
   * Test for inequality of two `padding_t`.
   *
   * Two `padding_t` are unequal iff any of their (corresponding) fields are
   * unequal.
   */
  bool operator!=(const padding_t &other) const;
} padding_t;

/**
 * Represents a vector in an image's coordinate space
 */
class ImageVect {
 public:
  /// A row coordinate (or row count) corresponding to the height dimension of
  /// an image (where index 0 starts at the top).
  int row;

  /// A column coordinate (or column count) corresponding to the width dimension
  /// of an image (where index 0 starts on the left).
  int col;

  /// A channel coordinate (or channel count) corresponding to the depth
  /// (considered to be non-spatial) dimension of an image.
  int channel;

  /**
   * Construct an image coordinate vector with the specified values
   */
  constexpr ImageVect(int const row, int const col, int const chan) noexcept
      : row(row), col(col), channel(chan) {}

  /**
   * Construct an image coordinate vector with the specified values.
   *
   * The order of elements in `coords` is row, column, channel.
   */
  ImageVect(const std::array<int, 3> coords) noexcept
      : row(coords[0]), col(coords[1]), channel(coords[2]) {}

  /**
   * Add another ImageVect to this one.
   */
  ImageVect operator+(ImageVect const &other) const;

  /**
   * Subtract another ImageVect from this one.
   */
  ImageVect operator-(ImageVect const &other) const;

  /**
   * Add some number of rows, columns and channels to this vector.
   */
  ImageVect add(int const rows, int const cols, int const chans = 0) const;

  /**
   * Subtract some number of rows, columns and channels from this vector.
   */
  ImageVect sub(int const rows, int const cols, int const chans = 0) const;

  /**
   * Test for equality of two ImageVects.
   *
   * Two ImageVects are considered equal iff all of their corresponding fields
   * are equal.
   */
  bool operator==(const ImageVect &other) const;

  /**
   * Test for inequality of two ImageVects.
   *
   * Two ImageVects are considered unequal iff any of their corresponding fields
   * are unequal.
   */
  bool operator!=(const ImageVect &other) const;
};

/**
 * Represents a rectangular sub-region of an image.
 */
class ImageRegion {
 public:
  struct {
    /// First row included in the region
    const int row;
    /// First column included in the region
    const int col;
    /// First channel included in the region
    const int channel;
  } start;

  struct {
    /// Number of rows included in the region (starting from `start.row`)
    const int height;
    /// Number of columns included in the region (starting from `start.col`)
    const int width;
    /// Number of channels included in the region (starting from
    /// `start.channel`)
    const int depth;
  } shape;

 public:
  /**
   * Construct an ImageRegion
   */
  constexpr ImageRegion(int row, int col, int chan, int height, int width,
                        int depth) noexcept
      : start{row, col, chan}, shape{height, width, depth} {}

  /**
   * Construct an ImageRegion
   */
  ImageRegion(const std::array<int, 3> start,
              const std::array<int, 3> shape) noexcept
      : start{start[0], start[1], start[2]},
        shape{shape[0], shape[1], shape[2]} {}

  /**
   * Get an ImageVect representing the start coordinate of this image region.
   */
  ImageVect StartVect() const;

  /**
   * Get an ImageVect representing the end coordinate of this image region.
   *
   * If `inclusive` is `true`, the coordinates given are the last considered to
   * be inside the region. Otherwise, the coordinates given are the smallest
   * considered to be after the region.
   */
  ImageVect EndVect(bool inclusive = false) const;

  /**
   * Test whether the specified coordinates are within this region.
   */
  bool Within(int row, int col, int channel) const;

  /**
   * The number of pixels within this region.
   */
  int PixelCount() const;

  /**
   * The number of elements within this region.
   */
  int ElementCount() const;

  /**
   * Determine the number of channel output groups that this region spans.
   *
   * `output_channels_per_group` is the number of channels per output channel
   * group.
   */
  int ChannelOutputGroups(int output_channels_per_group) const;
};

inline std::ostream &operator<<(std::ostream &stream, const padding_t &pad) {
  return stream << "(" << pad.top << "," << pad.left << "," << pad.bottom << ","
                << pad.right << ")";
}

inline std::ostream &operator<<(std::ostream &stream, const ImageVect &vect) {
  return stream << "(" << vect.row << "," << vect.col << "," << vect.channel
                << ")";
}

inline std::ostream &operator<<(std::ostream &stream, const ImageRegion &r) {
  const auto end = r.EndVect();
  return stream << "{ [" << r.start.row << "," << end.row << "), "
                << "[" << r.start.col << "," << end.col << "), "
                << "[" << r.start.channel << "," << end.channel << ") }";
}
}  // namespace nn
