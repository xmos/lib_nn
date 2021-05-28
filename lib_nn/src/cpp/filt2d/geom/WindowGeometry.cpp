
#include "geom/WindowGeometry.hpp"

using namespace nn;

ImageVect WindowGeometry::WindowOffset(const ImageVect &output_coords) const {
  auto out_row = start.row + output_coords.row * stride.row;
  auto out_col = start.col + output_coords.col * stride.col;
  auto out_chan = output_coords.channel * stride.channel;

  return ImageVect(out_row, out_col, out_chan);
}

bool WindowGeometry::operator==(WindowGeometry other) const {
  return this->shape.height == other.shape.height &&
         this->shape.width == other.shape.width &&
         this->shape.depth == other.shape.depth &&
         this->start.row == other.start.row &&
         this->start.col == other.start.col &&
         this->stride.row == other.stride.row &&
         this->stride.col == other.stride.col &&
         this->stride.channel == other.stride.channel &&
         this->dilation.row == other.dilation.row &&
         this->dilation.col == other.dilation.col;
}
