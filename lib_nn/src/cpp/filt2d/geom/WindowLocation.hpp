#pragma once

#include "nn_types.h"
#include "../misc.hpp"
#include "../util/AddressCovector.hpp"
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

      // TODO: It would probably be good to add an iterator that iterates over the window elements.
      //       It would also be nice to have that for the kernel tensor, 
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


}