
#include "ImageGeometry.hpp"

using namespace nn::filt2d;

mem_stride_t geom::ImageGeometry::getStride(
    const ImageVect& vect) const 
{ 
  return this->getStride(vect.row, vect.col, vect.channel); 
}


mem_stride_t geom::ImageGeometry::getStride(
    const int rows, 
    const int cols, 
    const int chans) const 
{ 
  return rows * this->rowBytes() + cols * this->pixelBytes() + chans * sizeof(T_elm); 
}

mem_stride_t geom::ImageGeometry::getStride(
    const ImageVect& from, 
    const ImageVect& to) const
{
  return this->getStride(to.row-from.row, to.col-from.col, to.channel-from.channel);
}

AddressCovector<geom::ImageGeometry::T_elm> geom::ImageGeometry::getAddressCovector() const 
{ 
  return AddressCovector<T_elm>(width, depth);
}

  
bool geom::ImageGeometry::operator==(
    ImageGeometry other) const 
{
  return this->height == other.height
      && this->width  == other.width 
      && this->depth  == other.depth;
}