#include "geom/Filter2dGeometry.hpp"

using namespace nn;

bool Filter2dGeometry::operator==(Filter2dGeometry other) const {
  return this->input == other.input && this->output == other.output &&
         this->window == other.window;
}

bool Filter2dGeometry::operator!=(Filter2dGeometry other) const {
  return !(*this == other);
}

WindowLocation Filter2dGeometry::GetWindow(
    const ImageVect output_coords) const {
  return WindowLocation(*this, output_coords);
}

WindowLocation Filter2dGeometry::GetWindow(const int row, const int col,
                                           const int channel) const {
  return GetWindow(ImageVect(row, col, channel));
}

padding_t Filter2dGeometry::Padding() const {
  padding_t padding;
  padding.top = -window.start.row;
  padding.left = -window.start.col;

  auto t = window.start.row + (output.height - 1) * window.stride.row +
           (window.shape.height - 1) * window.dilation.row;
  auto s = window.start.col + (output.width - 1) * window.stride.col +
           (window.shape.width - 1) * window.dilation.col;

  padding.bottom = t - (input.height - 1);
  padding.right = s - (input.width - 1);

  padding.MakeUnsigned();
  return padding;
}

bool Filter2dGeometry::IsDepthwise() const {
  // A model is depthwise if the window channel stride is 1 and its depth is 1.
  // If stride is 0, then it is 'dense'. If stride is not 0 or 1, or if the
  // stride is 1 and the depth is not 0, the behavior is undefined.
  return window.stride.channel == 1 && window.shape.depth == 1;
}
