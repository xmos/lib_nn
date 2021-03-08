#pragma once


namespace nn {
namespace filt2d {
namespace geom {

template <typename T_elm_in = int8_t>
class WindowGeometry {

  public:

    struct {
      unsigned const height;
      unsigned const width;
      unsigned const depth;
    } shape;

    struct {
      int const row;
      int const col;
    } start;

    struct {
      int const row;    
      int const col;    
      int const channel;
    } stride;

    struct {
      int const row;
      int const col;
    } dilation;


  public:

    WindowGeometry(
      unsigned const height,
      unsigned const width,
      unsigned const depth,
      int const start_row = 0,
      int const start_col = 0,
      int const stride_rows = 1,
      int const stride_cols = 1,
      int const stride_chans = 0,
      int const dil_rows = 1,
      int const dil_cols = 1)
        : shape{height, width, depth}, 
          start{start_row, start_col}, 
          stride{stride_rows, stride_cols, stride_chans}, 
          dilation{dil_rows, dil_cols} {}

    AddressCovector<T_elm_in> getPatchAddressCovector() const;

    unsigned pixelBytes() const;
    unsigned rowBytes() const;
    unsigned windowBytes() const;

    
    unsigned pixelElements() const;
    unsigned rowElements() const;
    unsigned windowElements() const;
};


template <typename T>
AddressCovector<T> WindowGeometry<T>::getPatchAddressCovector() const
  { return AddressCovector<T>(rowBytes(), pixelBytes(), sizeof(T)); }


template <typename T>
unsigned WindowGeometry<T>::pixelElements() const { return this->shape.depth; }
template <typename T>
unsigned WindowGeometry<T>::rowElements() const { return this->shape.width * this->pixelElements(); }
template <typename T>
unsigned WindowGeometry<T>::windowElements() const { return this->shape.height * this->rowElements(); }


template <typename T>
unsigned WindowGeometry<T>::pixelBytes() const { return this->pixelElements() * sizeof(T); }
template <typename T>
unsigned WindowGeometry<T>::rowBytes() const { return this->rowElements() * sizeof(T); }
template <typename T>
unsigned WindowGeometry<T>::windowBytes() const { return this->windowElements() * sizeof(T); }



}}}