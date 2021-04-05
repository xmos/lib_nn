
#include "WindowLocation.hpp"

#include <cassert>

using namespace nn;


ImageVect WindowLocation::InputStart() const
{
  return filter.window.WindowCoords(this->output_coords);
}

ImageVect WindowLocation::InputEnd() const
{
  return  InputStart().add(filter.window.shape.height - 1,
                           filter.window.shape.width  - 1,
                           filter.window.shape.depth  - 1);
}

ImageVect WindowLocation::InputCoords(const int filter_row,
                                      const int filter_col,
                                      const int filter_chan) const
{ 
  assert(filter_row >= 0);
  assert(filter_row < filter.window.shape.height);
  assert(filter_col >= 0);
  assert(filter_col < filter.window.shape.width);
  assert(filter_chan >= 0);
  assert(filter_chan < filter.window.shape.depth);

  const auto in_start = this->InputStart();

  return in_start.add(filter_row * filter.window.dilation.row,
                      filter_col * filter.window.dilation.col,
                      filter_chan);
}

padding_t WindowLocation::Padding() const
{
  const auto first_pix = InputStart();
  const auto last_pix = InputEnd();

  const int X_h = filter.input.height;
  const int X_w = filter.input.width;
  const int K_h = filter.window.shape.height;
  const int K_w = filter.window.shape.width;

  padding_t res;

  if( (last_pix.row < 0) || (last_pix.col < 0) || (first_pix.row >= X_h) || (first_pix.col >= X_w) ){
    // If any of those conditions are met, the window is entirely outside the input image.
    res.top = K_h;
    res.left = K_w;
    res.bottom = 0;
    res.right = 0;
  } else {
    // When dilation is 1x1, the computation is easy..
    res.top    = std::max<int>( -first_pix.row, 0 );
    res.left   = std::max<int>( -first_pix.col, 0 );
    res.bottom = std::max<int>( last_pix.row - (X_h - 1), 0);
    res.right  = std::max<int>( last_pix.col - (X_w - 1), 0);
    if( filter.window.UsesDilation() ){
      //If dilation is used, these need to be modified. To get the correct values in this case, we divide the padding
      // we get assuming 1x1 dilation by the dilation, always rounding up. (note: the 'padding assuming 1x1 dilation'
      // isn't *actually* assuming 1x1 dilation because the dilation was used to compute last_pix)
      res.top    = (res.top    + K_h - 1) / filter.window.dilation.row;
      res.left   = (res.left   + K_w - 1) / filter.window.dilation.col;
      res.bottom = (res.bottom + K_h - 1) / filter.window.dilation.row;
      res.right  = (res.right  + K_w - 1) / filter.window.dilation.col;
    }
  }

  return res;
}

bool WindowLocation::IsPadding(const int filter_row,
                               const int filter_col,
                               const int filter_chan) const
{
  auto coords = InputCoords(filter_row, filter_col, filter_chan);
  return !filter.input.IsWithinImage(coords);

}