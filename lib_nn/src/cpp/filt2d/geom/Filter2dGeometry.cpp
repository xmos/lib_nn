
#include "Filter2dGeometry.hpp"

#include "WindowLocation.hpp"

using namespace nn;



const ImageRegion Filter2dGeometry::GetFullJob() const 
{ 
  return ImageRegion(0,0,0, output.height, output.width, output.depth); 
}



bool Filter2dGeometry::operator==(
    Filter2dGeometry other) const 
{      
  return this->input  == other.input
      && this->output == other.output
      && this->window == other.window;
}



bool Filter2dGeometry::operator!=(
    Filter2dGeometry other) const 
{      
  return !(*this == other);
}


WindowLocation Filter2dGeometry::GetWindow(const ImageVect output_coords) const
{
  return WindowLocation(*this, output_coords);
}

WindowLocation Filter2dGeometry::GetWindow(const int row,
                                           const int col,
                                           const int channel) const
{
  return GetWindow(ImageVect(row, col, channel));
}


bool Filter2dGeometry::ModelIsDepthwise() const 
{
  // A model is depthwise if the window channel stride is 1 and its depth is 1.
  // If stride is 0, then it is 'dense'. If stride is not 0 or 1, or if the 
  // stride is 1 and the depth is not 0, the behavior is undefined.
  return window.stride.channel == 1 && window.shape.depth == 1;
}



padding_t Filter2dGeometry::ModelPadding(
      bool initial,
      bool signed_padding) const
{    
  padding_t pad { 
    .top    = int16_t( -window.start.row ),
    .left   = int16_t( -window.start.col ),
    .bottom = int16_t( (window.start.row + window.shape.height) - input.height ),
    .right  = int16_t( (window.start.col + window.shape.width ) - input.width  )};

  if(!initial){
    pad.top    -= (output.height - 1) * window.stride.row;
    pad.left   -= (output.width  - 1) * window.stride.col;
    pad.bottom += (output.height - 1) * window.stride.row;
    pad.right  += (output.width  - 1) * window.stride.col;
  }

  if(!signed_padding)
    pad.MakeUnsigned();

  return pad;
}



bool Filter2dGeometry::ModelRequiresPadding() const {
  return ModelPadding(true, true).HasPadding() || ModelPadding(false,true).HasPadding();
}



bool Filter2dGeometry::ModelFilterWindowAlwaysIntersectsInput() const {
  // If the convolution window ever entirely leaves the input image, the padding count for 
  // either the top-left or bottom-right (initial and final padding) will meet or exceed the 
  // window size for one of the dimensions
  int K_h = window.shape.height;
  int K_w = window.shape.width;

  auto pad = ModelPadding(true, true);
  if(pad.top  >= K_h || pad.bottom >= K_h) return false;
  if(pad.left >= K_w || pad.right  >= K_w) return false;
  pad = ModelPadding(false, true);
  if(pad.top  >= K_h || pad.bottom >= K_h) return false;
  if(pad.left >= K_w || pad.right  >= K_w) return false;

  return true;
}



// bool Filter2dGeometry::ModelConsumesInput() const
// {
//   // If the filter is depthwise (e.g. window.stride.channel == 1), then the input and output
//   // images must have the same number of channels. Otherwise we don't need to worry about channels.
//   if(ModelIsDepthwise() && input.depth != output.depth)
//     return false;

//   // Next, determine the minimum and maximum coordinate values used from the input image,
//   // even if they're outside the image.
//   const int row_min = window.start.row;
//   const int col_min = window.start.col;

//   int row_max = window.start.row + (output.height - 1) * window.stride.row 
//                                  + (window.shape.height - 1) * window.dilation.row;
//   int col_max = window.start.col + (output.width - 1) * window.stride.col
//                                  + (window.shape.width - 1) * window.dilation.col;

//   if(window.dilation.row == 1 && window.dilation.col == 1){
//     // If there's no dilation, we mostly just need to know that the minimim coordinates are <= 0 and
//     // the maximum coordinates are >= X dimension - 1, and that the stride is not larger than
//     // the window size (because then there would be skipped pixels between strides).
    
//     if(  window.stride.row > window.shape.height 
//       || row_min > 0 
//       || row_max > (input.height - 1)) return false;

//     if(  window.stride.col > window.shape.width
//       || col_min > 0
//       || col_max > (input.width - 1)) return false;

//     return true;

//   } else {
//     // Otherwise.. well, if the stride and dilation are both even, if definitely does not
//     //  consume the whole input.
//     if((window.stride.row % 2) == 0 && (window.dilation.row % 2) == 0)
//       return false;
//     if((window.stride.col % 2) == 0 && (window.dilation.col % 2) == 0)
//       return false;

//     // otherwise.. I'm not sure yet how to check it, show of iterating over each input element.
//     assert(0);
//     return false;
//   }
// }