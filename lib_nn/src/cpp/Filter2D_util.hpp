#pragma once

#include <type_traits>

namespace nn {

template <typename T>
class ImageVect {

  public:

    const T row;
    const T col;
    const T channel;

    ImageVect(
      T const img_row,
      T const img_col,
      T const img_chan)
        : row(img_row), col(img_col), channel(img_chan){}

    ImageVect<T> operator+(ImageVect<T> const& other){
      return this->add(other.row, other.col, other.chans);
    }

    ImageVect<T> operator-(ImageVect<T> const& other){
      return this->sub(other.row, other.col, other.chans);
    }

    ImageVect<T> add(T const rows, T const cols, T const chans)
    {
      return ImageVect<T>(this->row + rows, this->col + cols, this->channel + chans);
    }

    ImageVect<T> sub(T const rows, T const cols, T const chans)
    {
      return ImageVect<T>(this->row - rows, this->col - cols, this->channel - chans);
    }
};


}