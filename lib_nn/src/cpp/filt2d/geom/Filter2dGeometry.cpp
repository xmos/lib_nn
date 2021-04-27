
#include "geom/Filter2dGeometry.hpp"

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


padding_t Filter2dGeometry::Padding() const
{
  padding_t padding;
  padding.top = -window.start.row;
  padding.left = -window.start.col;

  auto tmp = GetWindow(output.height-1, output.width-1, 0).Padding();
  padding.bottom = tmp.bottom;
  padding.right = tmp.right;
  return padding;
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

