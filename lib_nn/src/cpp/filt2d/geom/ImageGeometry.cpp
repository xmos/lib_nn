
#include "geom/ImageGeometry.hpp"

using namespace nn;

mem_stride_t ImageGeometry::getStride(
    const ImageVect& vect) const 
{ 
  return this->getStride(vect.row, vect.col, vect.channel); 
}


mem_stride_t ImageGeometry::getStride(
    const int rows, 
    const int cols, 
    const int chans) const 
{ 
  return rows * this->rowBytes() + cols * this->pixelBytes() + chans * channel_depth; 
}

mem_stride_t ImageGeometry::getStride(
    const ImageVect& from, 
    const ImageVect& to) const
{
  return this->getStride(to.row-from.row, to.col-from.col, to.channel-from.channel);
}

  
bool ImageGeometry::operator==(
    ImageGeometry other) const 
{
  return this->height == other.height
      && this->width  == other.width 
      && this->depth  == other.depth
      && this->channel_depth == other.channel_depth;
}


bool ImageGeometry::IsWithinImage(const ImageVect& coords) const 
{
  return IsWithinImage(coords.row, coords.col, coords.channel);
}

bool ImageGeometry::IsWithinImage(const int row, 
                                  const int col, 
                                  const int channel) const
{
  if(row < 0 || col < 0 || channel < 0) return false;
  if(row >= height || col >= width || channel >= depth) return false;
  return true;
}