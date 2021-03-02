

#include "Filter2dGeometry.hpp"


using namespace nn::filt2d;



template <typename T_input, typename T_output>
InputCoordTransform geom::Filter2dGeometry<T_input,T_output>::GetInputCoordTransform()
{
  return {
    { this->window.start.row,   this->window.start.col,   this->window.start.channel   },
    { this->window.stride.rows, this->window.stride.cols, this->window.stride.channels },
  };
}


template <typename T_input, typename T_output>
PaddingTransform geom::Filter2dGeometry<T_input,T_output>::GetPaddingTransform() const
{
  return PaddingTransform(
      - this->window.start.row, this->window.start.col,
      (this->window.shape.height - this->input.height) + this->window.start.row,
      (this->window.shape.width - this->input.width) + this->window.start.col,
      - this->window.stride.row, - this->window.stride.col,
      this->window.stride.row, this->window.stride.col);
}


//astew: I hate that I have to choose between this and putting all definitions in header files -_-
//       Especially since this function is the exact same regardless of the template parameters..
template PaddingTransform nn::filt2d::geom::Filter2dGeometry<int8_t,int8_t>::GetPaddingTransform() const;