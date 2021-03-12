
#include "AddressCovector.hpp"

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


