#pragma once

#include <cassert>
#include <cstdlib>

#include "ImageGeometry.hpp"
#include "WindowGeometry.hpp"
#include "util.hpp"

namespace nn {

class WindowLocation;

/**
 * Filter2dGeometry is an abstraction representing the geometry of a 2D image
 * filter operation.
 *
 * Such an operation is described by three parts, the geometry of the filter's
 * output image, the geometry of the filter's input image, and the geometry of
 * the filter's window.
 *
 * Each output element Y[row,col,chan] is the result of some operation (e.g.
 * convolution or pooling) applied to a receptive field from the input image.
 * The shape of the receptive field is constant for all output pixel locations,
 * and is described by a (regular) 2D (spatial) grid placed with some (2D)
 * offset relative to the top-left pixel of the input image (X[0,0]).
 *
 * For output Y[0,0], the offset of the receptive field is given by the window
 * geometry's `start` (e.g. window.start = (0,0) indicates the window
 * corresponding to Y[0,0] starts aligned with the top-left edge of the input
 * image). The offset of the receptive field changes linearly as a function of
 * the output pixel coordinates, as given by `window.start` and `window.stride`.
 *
 * This class works for both depthwise filters and "dense" filters.
 *
 * The field `window.shape.depth` is the number of input channels that each
 * output channel is receptive to. Thus, for a depthwise filter,
 * `window.shape.depth` must be 1, and for a "dense" filter,
 * `window.shape.depth` should equal `input.depth`.
 */
class Filter2dGeometry {
 public:
  /**
   * The geometry of the filter's input image.
   */
  ImageGeometry input;

  /**
   * The geometry of the filter's output image.
   */
  ImageGeometry output;

  /**
   * The geometry of the filter's window
   */
  WindowGeometry window;

 public:
  /**
   * Default constructor, contents undefined.
   */
  constexpr Filter2dGeometry() noexcept : input(), output(), window() {}

  /**
   * Creates a filter geometry using the specified component geometries.
   */
  constexpr Filter2dGeometry(ImageGeometry input_geom,
                             ImageGeometry output_geom,
                             WindowGeometry window_geom) noexcept
      : input(input_geom), output(output_geom), window(window_geom) {}

  /**
   * Test for equality of two filter geometries.
   *
   * Two filter geometries are equal iff each of their component geometries
   * (input, output, window) are equal.
   */
  bool operator==(Filter2dGeometry other) const;

  /**
   * Test for inequality of two filter geometries.
   *
   * Two filter geometries are unequal iff any of their component geometries
   * (input, output, window) are unequal.
   */
  bool operator!=(Filter2dGeometry other) const;

  /**
   * Get an object representing this filter geometry with the filter window
   * bound to its receptive field for the specified output element.
   */
  WindowLocation GetWindow(const ImageVect output_coords) const;

  /**
   * Get an object representing this filter geometry with the filter window
   * bound to its receptive field for the specified output element.
   */
  WindowLocation GetWindow(const int row, const int col,
                           const int channel) const;

  /**
   * Returns true iff this geometry describes a standard depthwise filter.
   */
  bool IsDepthwise() const;

  /**
   * Get the implied padding of this filter geometry.
   *
   * This is the amount of padding that would need to be added around the input
   * image to avoid having the receptive field ever extend beyond the bounds of
   * the input image.
   *
   * Note that this does NOT give the same information as
   * WindowLocation::Padding() when the filter window has dilation values other
   * than 1. WindowLocation::Padding() indicates the number of rows/columns _of
   * the receptive field itself_ which are beyond the input image's bounds (i.e.
   * the "gaps" due to dilation do not count). This function indicates the
   * padding required if the input image is to _actually be padded in memmory_.
   */
  padding_t Padding() const;
};

/////////////////////////
inline std::ostream &operator<<(std::ostream &stream,
                                const Filter2dGeometry &filt) {
  return stream << "input{" << filt.input << "}, output{" << filt.output
                << "}, window{" << filt.window << "}";
}

}  // namespace nn

#include "WindowLocation.hpp"
