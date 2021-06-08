
#include "geom/WindowLocation.hpp"

#include <cassert>

using namespace nn;

ImageVect WindowLocation::InputStart() const {
  return filter.window.WindowOffset(this->output_coords);
}

ImageVect WindowLocation::InputEnd() const {
  return InputStart().add(
      (filter.window.shape.height - 1) * filter.window.dilation.row,
      (filter.window.shape.width - 1) * filter.window.dilation.col,
      filter.window.shape.depth - 1);
}

ImageVect WindowLocation::InputCoords(const int filter_row,
                                      const int filter_col,
                                      const int filter_chan) const {
  assert(filter_row >= 0);
  assert(filter_row < filter.window.shape.height);
  assert(filter_col >= 0);
  assert(filter_col < filter.window.shape.width);
  assert(filter_chan >= 0);
  assert(filter_chan < filter.window.shape.depth);

  const auto in_start = this->InputStart();

  return in_start.add(filter_row * filter.window.dilation.row,
                      filter_col * filter.window.dilation.col, filter_chan);
}

int WindowLocation::InputIndex(const int filter_row, const int filter_col,
                               const int filter_chan) const {
  assert(filter_row >= 0);
  assert(filter_row < filter.window.shape.height);
  assert(filter_col >= 0);
  assert(filter_col < filter.window.shape.width);
  assert(filter_chan >= 0);
  assert(filter_chan < filter.window.shape.depth);

  if (this->IsPadding(filter_row, filter_col, filter_chan)) return -1;

  return this->filter.input.Index(
      InputCoords(filter_row, filter_col, filter_chan));
}

padding_t WindowLocation::Padding() const {
  auto r = SignedPadding();
  r.MakeUnsigned();
  return r;
}

padding_t WindowLocation::SignedPadding() const {
  const auto first_pix = InputStart();
  const auto last_pix = InputEnd();

  const int X_h = filter.input.height;
  const int X_w = filter.input.width;
  const int K_h = filter.window.shape.height;
  const int K_w = filter.window.shape.width;

  padding_t res;

  if ((last_pix.row < 0) || (last_pix.col < 0) || (first_pix.row >= X_h) ||
      (first_pix.col >= X_w)) {
    // If any of those conditions are met, the window is entirely outside the
    // input image.
    res.top = K_h;
    res.left = K_w;
    res.bottom = 0;
    res.right = 0;
  } else if (!filter.window.UsesDilation()) {
    // When dilation is 1x1..
    res.top = -first_pix.row;
    res.left = -first_pix.col;
    res.bottom = last_pix.row - (X_h - 1);
    res.right = last_pix.col - (X_w - 1);
  } else {
    res.top = (-first_pix.row + filter.window.dilation.row - 1) /
              filter.window.dilation.row;
    res.left = (-first_pix.col + filter.window.dilation.col - 1) /
               filter.window.dilation.col;
    res.bottom = (last_pix.row - (X_h - 1) + filter.window.dilation.row - 1) /
                 filter.window.dilation.row;
    res.right = (last_pix.col - (X_w - 1) + filter.window.dilation.col - 1) /
                filter.window.dilation.col;
  }

  return res;
}

bool WindowLocation::IsPadding(const int filter_row, const int filter_col,
                               const int filter_chan) const {
  auto coords = InputCoords(filter_row, filter_col, filter_chan);
  return !filter.input.IsWithinImage(coords);
}