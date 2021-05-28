#include "geom/ImageGeometry.hpp"

using namespace nn;

int ImageGeometry::Index(const int row, const int col,
                         const int channel) const {
  if (row < 0 || row >= this->height) return -1;
  if (col < 0 || col >= this->width) return -1;
  if (channel < 0 || channel >= this->depth) return -1;

  return this->depth * (this->width * row + col) + channel;
}

int ImageGeometry::Index(const ImageVect &input_coords) const {
  return this->Index(input_coords.row, input_coords.col, input_coords.channel);
}

mem_stride_t ImageGeometry::GetStride(const ImageVect &vect) const {
  return this->GetStride(vect.row, vect.col, vect.channel);
}

mem_stride_t ImageGeometry::GetStride(const int rows, const int cols,
                                      const int chans) const {
  return rows * this->RowBytes() + cols * this->PixelBytes() +
         chans * channel_depth;
}

mem_stride_t ImageGeometry::GetStride(const ImageVect &from,
                                      const ImageVect &to) const {
  return this->GetStride(to.row - from.row, to.col - from.col,
                         to.channel - from.channel);
}

bool ImageGeometry::operator==(ImageGeometry other) const {
  return this->height == other.height && this->width == other.width &&
         this->depth == other.depth &&
         this->channel_depth == other.channel_depth;
}

bool ImageGeometry::IsWithinImage(const ImageVect &coords) const {
  return IsWithinImage(coords.row, coords.col, coords.channel);
}

bool ImageGeometry::IsWithinImage(const int row, const int col,
                                  const int channel) const {
  if (row < 0 || col < 0 || channel < 0) return false;
  if (row >= height || col >= width || channel >= depth) return false;
  return true;
}