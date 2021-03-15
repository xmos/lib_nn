
#include "WindowGeometry.hpp"

using namespace nn::filt2d::geom;

nn::filt2d::AddressCovector<WindowGeometry::T_elm_in> WindowGeometry::getPatchAddressCovector() const
{ 
  return AddressCovector<T_elm_in>(shape.width, shape.depth); 
}


unsigned WindowGeometry::pixelElements() const 
{ 
  return this->shape.depth; 
}
unsigned WindowGeometry::rowElements() const 
{ 
  return this->shape.width * this->pixelElements(); 
}
unsigned WindowGeometry::windowElements() const 
{ 
  return this->shape.height * this->rowElements(); 
}


unsigned WindowGeometry::pixelBytes() const 
{ 
  return this->pixelElements() * sizeof(T_elm_in); 
}
unsigned WindowGeometry::rowBytes() const 
{ 
  return this->rowElements() * sizeof(T_elm_in); 
}
unsigned WindowGeometry::windowBytes() const 
{ 
  return this->windowElements() * sizeof(T_elm_in); 
}

unsigned WindowGeometry::windowPixels() const 
{ 
  return this->shape.height * this->shape.width; 
}

bool WindowGeometry::operator==(WindowGeometry other) const {
  return this->shape.height == other.shape.height
      && this->shape.width  == other.shape.width
      && this->shape.depth  == other.shape.depth
      && this->start.row == other.start.row
      && this->start.col == other.start.col
      && this->stride.row == other.stride.row
      && this->stride.col == other.stride.col
      && this->stride.channel == other.stride.channel
      && this->dilation.row == other.dilation.row
      && this->dilation.col == other.dilation.col;
}
