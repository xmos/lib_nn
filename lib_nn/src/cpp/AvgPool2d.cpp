
#include "AvgPool2d.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace nn;

static int ceil_log2(uint32_t a)
{
    if(a == 0) return -1;
#ifdef  __XS3A__
    unsigned x;
    asm("clz %0, %1" : "=r"(x) : "r"(a));
    unsigned y = 31-x;

    //  clz(1) = 31 -> 31-31 = 0 -> 2^0 = 1
    //  clz(2) = 30 -> 31-30 = 1 -> 2^1 = 2
    //  clz(3) = 30 -> 31-30 = 1 -> 2^1 = 2
    //      2^(y) <= a < 2^(y+1)
    //  check for the lower bound, which yields a different result
    if(a == (1<<y)) return y;
    return y+1;

#else
    for(unsigned i = 0; i < 31; i++){
        if((((unsigned)1)<<i) >= a){
            return i;
        }
    }
#endif
    return -1;
}

void AvgPool2d::ComputeScaleShift(const WindowGeometry& window,
                                  int8_t& input_scale,
                                  int16_t& output_shift)
{
  ComputeScaleShift(window.shape.imagePixels(), input_scale, output_shift);
}

void AvgPool2d::ComputeScaleShift(const int window_pixels,
                                  int8_t& input_scale,
                                  int16_t& output_shift)
{
  const int c = ceil_log2(uint32_t(window_pixels));

  assert(c != -1);

  if(window_pixels == (1<<c)){
    // window pixel count is already a power of 2
    input_scale = 1;
    output_shift = c;
  } else {
    const unsigned q = 31 - c - 6;

    // 2^31 / pix
    const unsigned g = 0x80000000 / window_pixels;
    const unsigned h = (g + (1 << (q-1))) >> q; //Rounding down-shift

    assert(h > (1<<6));
    assert(h < (1<<7));

    input_scale = (int8_t)h;
    output_shift = c+6;
  }
}


AvgPool2d_Generic::Params::Params( const Filter2dGeometry& filter,
                                   const ImageRegion& region,
                                   const int8_t padding_value )
    : ak_params(filter.output, region, ChannelsPerOutputGroup), 
      mem_params(filter, padding_value, ChannelsPerOutputGroup)
{
  int8_t scale;
  int16_t shift;
  ComputeScaleShift(filter.window, scale, shift);

  this->agg_params = AvgPoolPatchFn::Params(filter.window, scale);
  this->ot_params = ShiftInt8OutputTransform::Params(filter.output, shift);
}


AvgPool2d_Valid::Params::Params( const Filter2dGeometry& filter,
                                 const ImageRegion& region )
    : ak_params(filter.output, region, ChannelsPerOutputGroup), 
      mem_params(filter)
{
  int8_t scale;
  int16_t shift;
  ComputeScaleShift(filter.window, scale, shift);

  this->agg_params = AvgPoolDirectValidFn::Params(filter, scale);
  this->ot_params = ShiftInt8OutputTransform::Params(filter.output, shift);
}

bool AvgPool2d_Generic::SupportsGeometry(const Filter2dGeometry& filter)
{

  const auto& input = filter.input;
  const auto& output = filter.output;
  const auto& window = filter.window;

  /// TODO: Some basic sanity checks should probably be collected into a function elsewhere, like
  ///       making sure image shapes are positive.
  
  if( input.height <= 0 || input.width <= 0 || input.depth <= 0 || input.channel_depth <= 0) return false;
  if( output.height <= 0 || output.width <= 0 || output.depth <= 0 || output.channel_depth <= 0) return false;
  if( window.shape.height <= 0 || window.shape.width <= 0 || window.shape.depth <= 0 || window.shape.channel_depth <= 0) return false;


  // Input and output images must have the same number of channels.
  if( input.depth != output.depth ) return false;

  // And the elements must be the same size
  if( input.channel_depth != output.channel_depth ) return false;

  // Window depth and channel stride must each be exactly 1
  if( window.shape.depth != 1 || window.stride.channel != 1) return false;

  // Window channel depth must equal input chanel depth.
  if( window.shape.channel_depth != input.channel_depth ) return false;

  // Channel count must be a multiple of 4 to guarantee correct alignment
  if( input.depth % 4 != 0 ) return false;

  // There must be at least one pixel of the filter window intersecting
  // with the input image for every output pixel location.
  if( filter.GetWindow(0,0,0).IsPadding( window.shape.height-1, window.shape.width-1 ) ) return false;
  if( filter.GetWindow(output.height-1, output.width-1, 0).IsPadding( 0, 0 ) ) return false;

  // Otherwise, it's supported
  return true;
}

bool AvgPool2d_Valid::SupportsGeometry(const Filter2dGeometry& filter)
{

  // Geometries supported by MaxPool2d_Valid are a strict subset of those supported by MaxPool2d_Generic
  if( !AvgPool2d_Generic::SupportsGeometry( filter ) ) return false;

  const auto& input = filter.input;
  const auto& output = filter.output;
  const auto& window = filter.window;

  // Padding is not supported
  if( filter.Padding().HasPadding() ) return false;

  // Dilation other than 1 isn't supported
  if( window.dilation.row != 1 || window.dilation.col != 1 ) return false;

  // Otherwise, it's supported
  return true;
}


AvgPool2d_Generic::AvgPool2d_Generic(AbstractKernel::Params* ak_params,
                                     ImToColPadded* memcopy_handler,
                                     AvgPoolPatchFn* aggregate_handler,
                                     ShiftInt8OutputTransform* ot_handler,
                                     int8_t* scratch_mem)
    : Filter2D_DW( ak_params, memcopy_handler, aggregate_handler, 
                   ot_handler, scratch_mem, ChannelsPerOutputGroup)
{

}

AvgPool2d_Valid::AvgPool2d_Valid(AbstractKernel::Params* ak_params,
                                 DerefInputFn* memcopy_handler,
                                 AvgPoolDirectValidFn* aggregate_handler,
                                 ShiftInt8OutputTransform* ot_handler)
    : Filter2D_DW( ak_params, memcopy_handler, aggregate_handler, 
                   ot_handler, (int8_t*)nullptr, ChannelsPerOutputGroup)
{

}