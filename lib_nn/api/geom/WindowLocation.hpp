#pragma once

#include <cstdint>
#include <tuple>

#include "Filter2dGeometry.hpp"
#include "geom/util.hpp"
#include "nn_types.h"

namespace nn {

/**
 * Represents the geometry associated with a single output element of a
 * Filter2dGeometry.
 *
 * This is basically a Filter2dGeometry with the filter window bound to a
 * specific location over the input image. The location of the window is
 * determined by the output image element pointed to by `output_coords`.
 *
 * The usual way to obtain a WindowLocation object is to have a Filter2dGeometry
 * and call one of the Filter2dGeometry::GetWindow() overloads.
 *
 * The intention of this class is to simplify the user's interactions with an
 * input image by hiding the math required to transform between coordinate
 * systems and the logic associated with padding.
 *
 */
class WindowLocation {
 public:
  /// The geometry of the filter represented
  const Filter2dGeometry& filter;
  /// The output coordinates binding the location of the filter window.
  const ImageVect output_coords;

 public:
  /**
   * Construct a WindowLocation for the specified filter and output element.
   */
  WindowLocation(const Filter2dGeometry& filter, const ImageVect output_coords)
      : filter(filter), output_coords(output_coords) {}

  /**
   * The coordinates (in the input image's coordinate space) of the first
   * element of the filter window.
   *
   * The first element of the filter window is considered to be that at (row,
   * col, chan) = (0, 0, 0) in the filter window's internal coordinate system
   * (i.e. first row, first column, first channel).
   *
   * Note: If the bound filter window extends beyond the bounds of the input
   * image, the returned ImageVect may point outside the input image.
   */
  ImageVect InputStart() const;

  /**
   * The coordinates (in the input image's coordinate space) of the final
   * element of the filter window.
   *
   * The final element of the filter window is considered to be that at (row,
   * col, chan) = (filter.window.shape.height-1, filter.window.shape.width-1,
   * filter.window.shape.depth-1) in the filter window's internal coordinate
   * system (i.e. final row, final column, final channel).
   *
   * Note: If the bound filter window extends beyond the bounds of the input
   * image, the returned ImageVect may point outside the input image.
   */
  ImageVect InputEnd() const;

  /**
   * Transform coordinates from the filter window's internal coordinate system
   * to the input image's coordinate system.
   *
   * `(filter_row, filter_col, filter_chan)` refers to a window element in the
   * filter window's internal coordinate system, with the constraint:
   *
   *    0 <= filter_row  < filter.window.shape.height
   *    0 <= filter.col  < filter.window.shape.width
   *    0 <= filter.chan < filter.window.shape.depth
   *
   * The filter's window geometry, together with the output coordinates to which
   * this object is bound
   * (`output_coords`), uniquely determines the input image coordinates for each
   * element of the filter window.
   *
   * Note: This function uses assertions to verify that the constraints on
   * `filter_row`, `filter_col` and `filter_chan` mentioned above are satisfied.
   */
  ImageVect InputCoords(const int filter_row, const int filter_col,
                        const int filter_chan) const;

  /**
   * Get the input buffer index of the specified window element.
   *
   * `(filter_row, filter_col, filter_chan)` refers to a window element in the
   * filter window's internal coordinate system, with the constraint:
   *
   *    0 <= filter_row  < filter.window.shape.height
   *    0 <= filter.col  < filter.window.shape.width
   *    0 <= filter.chan < filter.window.shape.depth
   *
   * The filter's window geometry, together with the output coordinates to which
   * this object is bound
   * (`output_coords`), uniquely determines the input image coordinates for each
   * element of the filter window.
   *
   * The input buffer index is the index at which the specified window element
   * can be found in the input image, assuming that the input image is stored in
   * a flat memory buffer in the usual manner (i.e. with channel index changing
   * most rapidly, followed by column and finally row).
   *
   * If the filter window for the bound output coordinates extends beyond the
   * input image's bounds, the specified window element may be in the (implied)
   * padding around the input image. In that case, this function returns -1. The
   * caller should check the result before using it to look up an input element.
   *
   * Note: This function uses assertions to verify that the constraints on
   * `filter_row`, `filter_col` and `filter_chan` mentioned above are satisfied.
   */
  int InputIndex(const int filter_row, const int filter_col,
                 const int filter_chan) const;

  /**
   * Get the window padding values associated with the bound output coordinates.
   *
   * The window padding values are the number of rows or columns of the filter
   * window which extend beyond the bounds of the top, left, bottom and right
   * edges of the input image.
   *
   * For a filter which uses dilation values other than 1, this is NOT the same
   * as you would expect from Filter2dGeometry::Padding(), which provides the
   * _input image_ padding, rather than the _filter window_ padding.
   *
   * The input image padding indicates how far (in input image coordinates)
   * beyond each edge of the input image the filter will need to reach. The
   * filter window padding indicates how many rows or columns _inside the filter
   * window_ are filled with padding. When dilation values are greater than 1
   * these are not the same because the input elements corresponding to adjacent
   * window elements are not adjacent in the input image (i.e. dilation causes
   * the receptive field to skip over rows/cols of the input image)
   *
   * The padding returned is unsigned. With it, the number of window pixels that
   * are actually inside the bounds of the input image can be calculated as
   *
   *    auto pad = location.Padding();
   *    auto total_input_pixels = (filter.window.shape.height - (pad.top  +
   * pad.bottom))
   *                             *(filter.window.shape.width  - (pad.left +
   * pad.right ));
   *
   */
  padding_t Padding() const;

  /**
   * Get the signed window padding values associated with the bound output
   * coordinates.
   *
   * The window padding values are the number of rows or columns of the filter
   * window which extend beyond the bounds of the top, left, bottom and right
   * edges of the input image.
   *
   * For a filter which uses dilation values other than 1, this is NOT the same
   * as you would expect from Filter2dGeometry::Padding(), which provides the
   * _input image_ padding, rather than the _filter window_ padding.
   *
   * The input image padding indicates how far (in input image coordinates)
   * beyond each edge of the input image the filter will need to reach. The
   * filter window padding indicates how many rows or columns _inside the filter
   * window_ are filled with padding. When dilation values are greater than 1
   * these are not the same because the input elements corresponding to adjacent
   * window elements are not adjacent in the input image (i.e. dilation causes
   * the receptive field to skip over rows/cols of the input image)
   *
   * The padding returned is signed.
   *
   */
  padding_t SignedPadding() const;

  /**
   * Determine whether the specified window element is within the input image's
   * bounds.
   *
   * `(filter_row, filter_col, filter_chan)` refers to a window element in the
   * filter window's internal coordinate system, with the constraint:
   *
   *    0 <= filter_row  < filter.window.shape.height
   *    0 <= filter.col  < filter.window.shape.width
   *    0 <= filter.chan < filter.window.shape.depth
   */
  bool IsPadding(const int filter_row, const int filter_col,
                 const int filter_chan = 0) const;

  /**
   * Get a reference to the input image element corresponding to the specified
   * window element.
   *
   * `(filter_row, filter_col, filter_chan)` refers to a window element in the
   * filter window's internal coordinate system, with the constraint:
   *
   *    0 <= filter_row  < filter.window.shape.height
   *    0 <= filter.col  < filter.window.shape.width
   *    0 <= filter.chan < filter.window.shape.depth
   *
   * The reference returned is to the actual data element in the buffer to which
   * `input_image_base` points, and can be used to modify the input image data
   * itself. Because of this, the user must not specify a window element which
   * is outside the bounds of the input image.
   *
   * It is highly recommended, therefore, that IsPadding() be tested prior to
   * any call to this function. If just getting the effective input element
   * value (i.e. the input element value or the padding value) then the function
   * GetInput(), which doesn't return a reference, should be used instead.
   *
   * An assertion is made to guarantee the condition mentioned.
   *
   * `input_image_base` is the base address of the input image buffer. If a flat
   * array, `std::vector`, or similar container is used to manage the input
   * image buffer, then usually the appropriate argument is `&input_buffer[0]`.
   */
  template <typename T>
  T& InputElement(T* input_image_base, const int filter_row,
                  const int filter_col, const int filter_chan) const;

  /**
   * Get the effective input image element value corresponding to the specified
   * window element.
   *
   * `(filter_row, filter_col, filter_chan)` refers to a window element in the
   * filter window's internal coordinate system, with the constraint:
   *
   *    0 <= filter_row  < filter.window.shape.height
   *    0 <= filter.col  < filter.window.shape.width
   *    0 <= filter.chan < filter.window.shape.depth
   *
   * If the window element refers to a location beyond the bounds of the input
   * image's geometry, then `pad_value` is returned instead.
   *
   * `input_image_base` is the base address of the input image buffer. If a flat
   * array, `std::vector`, or similar container is used to manage the input
   * image buffer, then usually the appropriate argument is `&input_buffer[0]`.
   */
  template <typename T>
  T GetInput(const T* input_image_base, const int filter_row,
             const int filter_col, const int filter_chan,
             const T pad_value = 0) const;

  /**
   * Apply a fold operation across the window elements.
   *
   * The supplied callback will be called for each element of the filter window.
   * The signature and meaning of the callback function's arguments are as
   * indicated:
   *
   *      T_acc CallbackFunc(const ImageVect& filter_coords,
   *                         const ImageVect& input_coords,
   *                         const T_acc prev_accumulator,
   *                         const T_elm input_element,
   *                         const bool is_padding);
   *
   * filter_coords - row, column and channel indices of the window element (e.g.
   * (0,0,0) will always be first.) input_coords - The input image coordinates
   * corresponding to filter_coords prev_accumulator - The accumulator returned
   * from the previous call of the callback function (on the first call
   * initial_acc is passed in) input_element - The value of the input image
   * element corresponding to the current filter location. If the input_coords
   * are outside the input image, this will be pad_value. is_padding - Whether
   * the current input element is in the input's padding (i.e. outside the input
   * image)
   *
   * The window elements are traversed with indices in ascending order as in
   *
   *      for(int row = 0; row < window.height; row++)
   *        for(int col = 0; col < window.width; col++)
   *          for(int channel = 0; channel < window.depth; channel++)
   *            ...
   *
   *
   * `input_image_base` is the base address of the input image buffer. If a flat
   * array, `std::vector`, or similar container is used to manage the input
   * image buffer, then usually the appropriate argument is `&input_buffer[0]`.
   */
  template <typename T_acc, typename T_elm>
  T_acc Fold(const T_elm* input_image_base,
             std::function<T_acc(const ImageVect&, const ImageVect&,
                                 const T_acc, const T_elm, const bool)>
                 op,
             const T_acc initial_acc, const T_elm pad_value = 0);
};

/////////////////////////
template <typename T>
T& WindowLocation::InputElement(T* input_image_base, const int filter_row,
                                const int filter_col,
                                const int filter_chan) const {
  assert(!IsPadding(filter_row, filter_col, filter_chan));

  auto c =
      InputStart().add(filter_row * filter.window.dilation.row,
                       filter_col * filter.window.dilation.col, filter_chan);
  return filter.input.Element<T>(input_image_base, c.row, c.col, c.channel);
}

/////////////////////////
template <typename T>
T WindowLocation::GetInput(const T* input_image_base, const int filter_row,
                           const int filter_col, const int filter_chan,
                           const T pad_value) const {
  if (IsPadding(filter_row, filter_col, filter_chan)) return pad_value;

  return InputElement<T>(const_cast<T*>(input_image_base), filter_row,
                         filter_col, filter_chan);
}

/////////////////////////
template <typename T_acc, typename T_elm>
T_acc WindowLocation::Fold(
    const T_elm* input_image_base,
    std::function<T_acc(const ImageVect&, const ImageVect&, const T_acc,
                        const T_elm, const bool)>
        op,
    const T_acc initial_acc, const T_elm pad_value) {
  T_acc acc = initial_acc;

  for (int f_row = 0; f_row < this->filter.window.shape.height; ++f_row) {
    for (int f_col = 0; f_col < this->filter.window.shape.width; ++f_col) {
      for (int f_chan = 0; f_chan < this->filter.window.shape.depth; ++f_chan) {
        ImageVect filter_coords(f_row, f_col, f_chan);
        ImageVect input_coords = this->InputCoords(f_row, f_col, f_chan);

        bool is_padding = this->IsPadding(f_row, f_col, f_chan);
        T_elm val = this->GetInput<T_elm>(input_image_base, f_row, f_col,
                                          f_chan, pad_value);

        acc = op(filter_coords, input_coords, acc, val, is_padding);
      }
    }
  }

  return acc;
}

}  // namespace nn