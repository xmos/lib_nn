#pragma once

#include <array>
#include <cstdint>

#include "ImageGeometry.hpp"
#include "geom/util.hpp"
#include "nn_types.h"

namespace nn {

/**
 * Represents the geometry of the filter window (receptive field) in a 2D image
 * filtering operation (e.g. convolution or pooling).
 *
 * Note: "filter window", "convolution window", "pooling window" and "receptive
 * field" are all synonyms and used interchangeably for the purposes of this
 * documentation.
 *
 * For any given 2D filter operation, each output image element Y[,,] is
 * computed as a function of a subset of the input image elements X[,,], called
 * its receptive field W[,,]. The receptive field has two important conceptual
 * properties, its shape and its position.
 *
 * Note that there are three different coordinate systems involved here. The
 * input and output elements X[i,j,k] and Y[a,b,c] and in specified in the input
 * and output image coordinate systems respectively. The third coordinate system
 * is internal to the filter window, and is that implied when referring to an
 * element of the filter window W[r,s,t].
 *
 * The shape of the receptive field is identical for every Y[i,j,k], and is
 * determined by WindowGeometry::shape and WindowGeometry::dilation. `shape`
 * indicates the dimensions of the receptive field (in rows, columns and
 * channels). `dilation` (which only applies to the two spatial dimensions; not
 * channels/depth) indicates the vertical and horizontal "gaps" between the rows
 * and columns of the receptive field (channel gaps are not supported).
 *
 * The position of the receptive field is (usually) different for each Y[i,j,k],
 * although in a "dense" convolution, for example, the receptive field is
 * identical for all channels of each output pixel. The position of the
 * receptive field for output element Y[i,j,k] is given by the vector describing
 * W[0,0,0]'s (the top-left-lowest-channel element of the receptive field)
 * coordinates in the input image. In a Filter2dGeometry, the filter window's
 * position is a trilinear function of the output element coordinates, with
 * coefficients given in WindowGeometry::start and WindowGeometry::stride
 *
 * The receptive field of output element Y[0,0,0] has W[0,0,0] = X[start.row,
 * start.col, 0]. That is,
 * `(start.row, start.col)` is the spatial position (in the input image's
 * coordinate space) of the top-left pixel of the filter window for the top-left
 * output pixel (the start offset for channel is always zero). Negative values
 * for `start.row` indicate top padding, and negative values for `start.col`
 * indicate left padding.
 *
 * WindowGeometry::stride determines how changes in output element coordinates
 * effect changes in the filter window's position within the input image.
 * Specifically, `stride.row` is the number of rows the filter window moves down
 * (increasing row) in the input image for every row of the output image. e.g.
 *
 * if     Y[i,j,k] --> W[0,0,0] = X[p,q,r]
 * then   Y[i+d,j,k] --> W[0,0,0] = X[p + (d * stride.row), q, r]
 *
 * `stride.col` and `stride.channel` work similarly in the other two dimensions,
 * with the stipulation that `stride.channel` should be either 0, for "dense"
 * operations (i.e. where each output element is a function of ALL input
 * channels), or a 1 for "depthwise" operations.
 *
 *
 * ====
 *
 * Y[a,b,c] are the elements of the output image, with
 *    0 <= a < output.height
 *    0 <= b < output.width
 *    0 <= c < output.depth
 *
 * X[i,j,k] are the elements of the input image, with
 *    0 <= i < input.height
 *    0 <= j < input.width
 *    0 <= k < input.depth
 * (Additionally, filter operations which support padding extend X[i,j,k] to be
 * defined outside the spatial bounds of the input image (but not the channel
 * bounds).)
 *
 * W[r,s,t] are the elements of the receptive field of some output element
 * Y[a,b,c], with 0 <= r < window.shape.height 0 <= s < window.shape.width 0 <=
 * t < window.shape.depth Here, r, s and t are the internal coordinates of the
 * receptive field.
 *
 *
 * The mapping of window element `W[0,0,0]` onto the input image for output
 * element `Y[0,0,0]`:
 *
 *      W[0,0,0] --> X[start.row, start.col, 0]                   @ Y[0,0,0]
 *
 * The strides describe how the start of the receptive field moves with a unit
 * change in output coordinates:
 *
 *      W[0,0,0] --> X[start.row + stride.row,
 *                     start.col + stride.col,
 *                     0         + stride.channel]                @ Y[1,1,1]
 *
 * For an arbitrary output element `Y[a,b,c]`, the mapping of `W[0,0,0]` is then
 * given by:
 *
 *      W[0,0,0] --> X[start.row + a * stride.row,
 *                     start.col + b * stride.col,
 *                     0         + c * stride.channel]            @ Y[a,b,c]
 *
 *
 * Suppose that for output element `Y[a,b,c]`, `W[0,0,0]` maps onto input
 * element `X[i,j,k]`:
 *
 *      W[0,0,0] --> X[i,j,k]                                     @ Y[a,b,c]
 *
 * Dilation scales how the filter window samples the input image, with each axis
 * behaving independently:
 *
 *      W[1,1,1] --> X[i + dilation.row,
 *                     j + dilation..col,
 *                     k + 1]                                     @ Y[a,b,c]
 *
 *      W[r,s,t] --> X[i + r * dilation.row,
 *                     j + s * dilation.col,
 *                     k + t]                                     @ Y[a,b,c]
 *
 * But `X[i,j,k]` is just another name for `X[start.row, start.col, 0]`. So the
 * full description is:
 *
 *      W[r,s,t] --> X[start.row + a * stride.row     + r * dilation,
 *                     start.col + b * stride.col     + s * dilation,
 *                             0 + c * stride.channel + t *        1 ]    @
 * Y[a,b,c]
 *
 */
class WindowGeometry {
 public:
  /**
   * The dimensions of the filter window.
   *
   * shape.height and shape.width together describe the spatial size of the
   * filter window. shape.depth indicates the number of input image channels
   * used to compute each output image channel.
   *
   * Currently shape.depth should always either be equal to 1 (for a depthwise
   * filter2d operation), or should equal the number of channels in the input
   * image (for "dense" operations). Other values may be supported in the
   * future, but their behavior is currently undefined.
   *
   * NOTE: This is NOT the same as the patch geometry when using `MemCpyFn`s.
   * For "dense" operations it may be the same, but for depthwise operations
   * (e.g. average pooling or depthwise convolution), the patch geometry has a
   * depth given by the degree of parallelism of the operator (e.g. 16 for
   * average pool, 32 for max pool). Considerations for operator data
   * parallelism are outside the scope of this class.
   */
  ImageGeometry shape;

  /**
   * The anchor position of the filter window (in the input image's coordinate
   * space).
   *
   * For the top-left output element, the top-left element of the filter window
   * is the input element X[start.row, start.col, 0]
   */
  struct {
    /// Row (in the input image's coordinate space) of the filter window's
    /// anchor position.
    int row;
    /// Column (in the input image's coordinate space) of the filter window's
    /// anchor position.
    int col;
  } start;

  /**
   * The gradient of the receptive field's offset (in the input image's
   * coordinate space) with respect to the output element coordinates.
   */
  struct {
    /// Number of input rows the window shifts for each output row.
    int row;
    /// Number of input columns the window shifts for each output column
    int col;
    /// Number of input channels the window shifts for each output channel
    int channel;
  } stride;

  /**
   * Scales (spatially) how the receptive field samples elements from the input
   * image.
   *
   * dilation == (1,1) means the receptive field is a contiguous block of
   * pixels. Larger dilations mean the receptive field spans a larger portion of
   * the input image (while keeping the number of pixels in the receptive field
   * constant).
   */
  struct {
    /// Vertical spacing of samples
    int row;
    /// Horizontal spacing of samples
    int col;
  } dilation;

 public:
  /**
   * Default constructor.
   */
  constexpr WindowGeometry() noexcept
      : WindowGeometry(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0) {}

  /**
   * Construct a WindowGeometry with the specified values.
   */
  WindowGeometry(const std::array<int, 3> shape,
                 const std::array<int, 2> starts,
                 const std::array<int, 3> stride,
                 const std::array<int, 2> dilation,
                 int channel_depth = 1) noexcept
      : shape(shape[0], shape[1], shape[2], channel_depth),
        start{starts[0], starts[1]},
        stride{stride[0], stride[1], stride[2]},
        dilation{dilation[0], dilation[1]} {}

  /**
   * Construct a WindowGeometry with the specified values.
   */
  constexpr WindowGeometry(int const height, int const width, int const depth,
                           int const start_row = 0, int const start_col = 0,
                           int const stride_rows = 1, int const stride_cols = 1,
                           int const stride_chans = 0, int const dil_rows = 1,
                           int const dil_cols = 1,
                           int const channel_depth = 1) noexcept
      : shape(height, width, depth,
              channel_depth),  // asj: channel_depth could be bits_per_element
        start{start_row, start_col},
        stride{stride_rows, stride_cols, stride_chans},
        dilation{dil_rows, dil_cols} {}

  /**
   * Test equality between two window geometries.
   *
   * Two window geometries are equal iff all of their fields are equal.
   */
  bool operator==(WindowGeometry other) const;

  /**
   * Get the coordinates (in an input image's coordinate space) of the first
   * (i.e. top-left) pixel of the filter window corresponding to the specified
   * output element.
   */
  ImageVect WindowOffset(const ImageVect &output_coords) const;

  /**
   * Does the filter use a dilation other than 1?
   */
  inline bool UsesDilation() const {
    return (dilation.row != 1) || (dilation.col != 1);
  }
};

inline std::ostream &operator<<(std::ostream &stream,
                                const WindowGeometry &window) {
  return stream << "shape{" << window.shape.height << "," << window.shape.width
                << "," << window.shape.depth << "},"
                << "start{" << window.start.row << "," << window.start.col
                << "},"
                << "stride{" << window.stride.row << "," << window.stride.col
                << "," << window.stride.channel << "},"
                << "dilation{" << window.dilation.row << ","
                << window.dilation.col << "}";
}

}  // namespace nn
