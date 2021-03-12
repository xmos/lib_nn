#pragma once

#include "../Filter2d_util.hpp"
#include "ImageGeometry.hpp"
#include "WindowGeometry.hpp"

#include <cstdlib>




namespace nn {
namespace filt2d {
namespace geom {

template <typename T>
static inline T max(T left, T right){
  return (left >= right)? left : right;
}

template <typename T>
static inline T min(T left, T right){
  return (left <= right)? left : right;
}



template <typename T_input = int8_t, typename T_output = int8_t>
class Filter2dGeometry {

  public:

    ImageGeometry<T_input> const input;
    ImageGeometry<T_output> const output;
    WindowGeometry<T_input> const window;


  public:

    Filter2dGeometry(
      ImageGeometry<T_input> const input_geom,
      ImageGeometry<T_output> const output_geom,
      WindowGeometry<T_input> const window_geom)
        : input(input_geom), output(output_geom), window(window_geom) {}

    InputCoordTransform GetInputCoordTransform();
 
    PaddingTransform GetPaddingTransform() const;

    const ImageRegion GetFullJob() const { return ImageRegion(0,0,0, output.height, output.width, output.channels); }

    // These don't currently handle dilation correctly!!
    padding_t InitialPadding(bool get_signed = false) const {      
      padding_t pad { 
        .top    = int16_t( -window.start.row ),
        .left   = int16_t( -window.start.col ),
        .bottom = int16_t( (window.start.row + window.shape.height) - input.height ),
        .right  = int16_t( (window.start.col + window.shape.width ) - input.width  )};

      if(!get_signed) pad.makeUnsigned();

      return pad;
    }

    // These don't currently handle dilation correctly!!
    padding_t FinalPadding(bool get_signed = false) const {      
      padding_t pad = InitialPadding();

      pad.top    -= output.height * window.stride.row;
      pad.left   -= output.width  * window.stride.col;
      pad.bottom += output.height * window.stride.row;
      pad.right  += output.width  * window.stride.col;

      if(!get_signed) pad.makeUnsigned();

      return pad;
    }

};



namespace util {

  
static inline int32_t rand_int( int32_t min, int32_t max)
{
    uint32_t delta = max - min;
    uint32_t d = rand() % delta;
    return min + d;
}

static inline uint32_t rand_uint( uint32_t min,  uint32_t max)
{
    uint32_t delta = max - min;
    uint32_t d = rand() % delta;
    return min + d;
}


  template<typename T_elm>
  ImageGeometry<T_elm> RandomImageGeometry(
      unsigned min_height, unsigned max_height,
      unsigned min_width,  unsigned max_width,
      unsigned min_channels, unsigned max_channels,
      unsigned channel_step = 4)
  {

    const unsigned height = rand_uint(min_height, max_height);
    const unsigned width = rand_uint(min_width, max_width);
    const unsigned channels = min_channels + rand_uint(0, ((max_channels - min_channels / channel_step)) * channel_step);


    return ImageGeometry<T_elm>(height, width, channels);

  }

  // template<typename T_elm_in, typename T_elm_out>
  // Filter2dGeometry<T_elm_in,T_elm_out> RandomFilterGeometry()
  // {
  //   reuu
  // }


};


}}}