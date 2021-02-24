

#include "Filter2dGeometry.hpp"


using namespace nn::filt2d::geom;



template <typename T_input, typename T_output>
InputCoordTransform Filter2dGeometry<T_input,T_output>::GetInputCoordTransform()
{
  return {
    { this->window.start.row,   this->window.start.col,   this->window.start.channel   },
    { this->window.stride.rows, this->window.stride.cols, this->window.stride.channels },
  };
}


template <typename T_input, typename T_output>
PaddingTransform Filter2dGeometry<T_input,T_output>::GetPaddingTransform()
{
  return {
    .initial = {
      .top = - this->window.start.row,
      .left = this->window.start.col,
      .bottom = (this->window.shape.height - this->input.shape.height) + this->window.start.row,
      .right = (this->window.shape.width - this->input.shape.width) + this->window.start.col
    },

    .stride = {
      .top    = - this->window.stride.rows,
      .left   = - this->window.stride.cols,
      .bottom = this->window.stride.rows,
      .right  = this->window.stride.cols,
    }
  };
}

