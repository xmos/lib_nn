#pragma once

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "Filter2dGeometry.hpp"

#include <cstdint>
#include <tuple>

namespace nn {

  class WindowLocation {

    public:

      const Filter2dGeometry& filter;
      const ImageVect output_coords;

    public:

      WindowLocation(const Filter2dGeometry& filter,
                     const ImageVect output_coords)
        : filter(filter), output_coords(output_coords) {}

      /**
       * Get the coordinates for the first element (first channel of top-left pixel) of this window location,
       * in the input image's coordinate space.
       */
      ImageVect InputStart() const;

      /**
       * Get the coordinates for the final element (last channel of bottom-right pixel) of this window location,
       * in the input image's coordinate space.
       */
      ImageVect InputEnd() const;

      /**
       * Transform from filter window coordinates (`filter_row`, `filter_col`, `filter_chan`) to input image
       * coordinates.
       */
      ImageVect InputCoords(const int filter_row,
                            const int filter_col,
                            const int filter_chan) const;

      /**
       * Get the flattened index of the specified input element using local filter coordinates.
       * 
       * The "flattened" index of an element is the index of the element when the image is stored in a 1 dimensional
       * array. This is ideal, for example, when the input image is backed by a `std::vector` object.
       * 
       * This function returns -1 if the specified coordinates (at the current filter window location) refer to
       * an element in padding (i.e. beyond the bounds of the input image).
       */
      int InputIndex(const int filter_row,
                     const int filter_col,
                     const int filter_chan) const;

      /**
       * 
       */
      int FilterIndex(const int filter_row,
                      const int filter_col,
                      const int filter_chan) const;

      /**
       * Determine the top-, left-, bottom- and right-padding requirements for the filter window at this location.
       */
      padding_t Padding() const;

      bool IsPadding(const int filter_row,
                     const int filter_col,
                     const int filter_chan = 0) const;

      // TODO: This might need to change to InputElement() so that a KernelElement() can also be added.
      template <typename T>
      T& InputElement(T* input_image_base,
                      const int filter_row,
                      const int filter_col,
                      const int filter_chan) const;

      template <typename T>
      T GetInput(const T* input_image_base,
                 const int filter_row,
                 const int filter_col,
                 const int filter_chan,
                 const T pad_value = 0) const;

      /**
       * Apply a fold operation across the window elements.
       * 
       * The supplied callback will be called for each element within this window. The signature and meaning 
       * of the callback function's arguments are as indicated:
       * 
       * T_acc CallbackFunc(const ImageVect& filter_coords,
       *                    const ImageVect& input_coords,
       *                    const T_acc prev_accumulator,
       *                    const T input_element,
       *                    const bool is_padding);
       * 
       * filter_coords - row, column and channel indices of the window element (e.g. (0,0,0) will always be first.)
       * input_coords - The input image coordinates corresponding to filter_coords
       * prev_accumulator - The accumulator returned from the previous call of the callback function
       *                    (on the first call initial_acc is passed in)
       * input_element - The value of the input image element corresponding to the current filter location.
       *                 If the input_coords are in padding, this will be pad_value.
       * is_padding - Whether the current input element is in the input's padding (i.e. outside the input image)
       */
      template <typename T_acc, typename T_elm>
      T_acc Fold(const T_elm* input_image_base,
                 std::function<T_acc(const ImageVect&, 
                                     const ImageVect&, 
                                     const T_acc, 
                                     const T_elm, 
                                     const bool)> op,
                 const T_acc initial_acc,
                 const T_elm pad_value = 0);
  };


  template <typename T>
  T& WindowLocation::InputElement(T* input_image_base,
                                  const int filter_row,
                                  const int filter_col,
                                  const int filter_chan) const
  {
    assert( !IsPadding(filter_row, filter_col, filter_chan) );

    auto c = InputStart().add(filter_row * filter.window.dilation.row,
                              filter_col * filter.window.dilation.col,
                              filter_chan);
    return filter.input.Element<T>(input_image_base, c.row, c.col, c.channel);
  }

  template <typename T>
  T WindowLocation::GetInput(const T* input_image_base,
                             const int filter_row,
                             const int filter_col,
                             const int filter_chan,
                             const T pad_value) const
  {
    if( IsPadding(filter_row, filter_col, filter_chan) )
      return pad_value;
    
    return InputElement<T>( const_cast<T*>(input_image_base), filter_row, filter_col, filter_chan );
  }


  template <typename T_acc, typename T_elm>
  T_acc WindowLocation::Fold(const T_elm* input_image_base,
                             std::function<T_acc(const ImageVect&, 
                                                 const ImageVect&, 
                                                 const T_acc,
                                                 const T_elm, 
                                                 const bool)> op,
                             const T_acc initial_acc,
                             const T_elm pad_value)
  {

    T_acc acc = initial_acc;

    for(int f_row = 0; f_row < this->filter.window.shape.height; ++f_row){
      for(int f_col = 0; f_col < this->filter.window.shape.width; ++f_col){
        for(int f_chan = 0; f_chan < this->filter.window.shape.depth; ++f_chan){
          ImageVect filter_coords(f_row, f_col, f_chan);
          ImageVect input_coords = this->InputCoords(f_row, f_col, f_chan);

          bool is_padding = this->IsPadding(f_row, f_col, f_chan);
          T_elm val = this->GetInput<T_elm>(input_image_base, f_row, f_col, f_chan, pad_value);

          acc = op(filter_coords, input_coords, acc, val, is_padding);
        }
      }
    }

    return acc;
  }

}