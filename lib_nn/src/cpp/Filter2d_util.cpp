
#include "Filter2d_util.hpp"

using namespace nn::filt2d;


ImageVect ICoordinateConverter::getInputCoords(
  ImageVect const& output_coords) const
{
  auto const& transform = this->getTransform();

  return ImageVect(
    transform.start.row     + output_coords.row     * transform.stride.rows,
    transform.start.col     + output_coords.col     * transform.stride.cols,
    transform.start.channel + output_coords.channel * transform.stride.channels );
}


// template <typename T_input, typename T_output>
// void IIOPointerResolver<T_input, T_output>::resolvePointers(
//       T_output * output_img, 
//       T_input const* input_img,
//       ImageVect const& output_coords) const
// {
//   auto const& transform = this->getPointerTransform();

//   output_img = ((unsigned)output_img)
//              + transform.output.row_bytes  * output_coords.row
//              + transform.output.col_bytes  * output_coords.col
//              + transform.output.chan_bytes * output_coords.channel;

//   input_img  = ((unsigned)input_img)
//              + transform.input.row_bytes  * output_coords.row
//              + transform.input.col_bytes  * output_coords.col
//              + transform.input.chan_bytes * output_coords.channel;

// }


  padding_t const IPaddingResolver::getPadding(
      ImageVect const& output_coords,
      bool const get_unsigned) const
  {

    auto const& transform = this->getPaddingTransform();

    padding_t res = {
      .top    = (int16_t) (transform.initial.top    + transform.stride.top    * output_coords.row),
      .left   = (int16_t) (transform.initial.left   + transform.stride.left   * output_coords.col),
      .bottom = (int16_t) (transform.initial.bottom + transform.stride.bottom * output_coords.row),
      .right  = (int16_t) (transform.initial.right  + transform.stride.right  * output_coords.col),
    };

    if(get_unsigned){
      res.top    = (res.top    < 0)? 0 : res.top;
      res.left   = (res.left   < 0)? 0 : res.left;
      res.bottom = (res.bottom < 0)? 0 : res.bottom;
      res.right  = (res.right  < 0)? 0 : res.right;
    }

    return res;
  }