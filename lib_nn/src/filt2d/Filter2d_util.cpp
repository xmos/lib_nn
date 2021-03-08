
#include "Filter2d_util.hpp"

using namespace nn::filt2d;


int32_t AddressCovectorBase::dot(ImageVect coords) const
{
  return this->dot(coords.row, coords.col, coords.channel);
}

int32_t AddressCovectorBase::dot(int row, int col, int channel) const
{
  return row     * ((int32_t)this->row_bytes )
        + col     * ((int32_t)this->col_bytes )
        + channel * ((int32_t)this->chan_bytes);
}





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