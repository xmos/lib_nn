#pragma once

#include "../geom/util.hpp"

#include <cstdint>

namespace nn {
namespace filt2d {



class AddressCovectorBase {
  public:

    int16_t row_bytes;
    int16_t col_bytes;
    int16_t chan_bytes;
    int16_t const zero = 0;

  public:

    AddressCovectorBase(
      int16_t rowbytes, 
      int16_t colbytes, 
      int16_t chanbytes)
        : row_bytes(rowbytes), col_bytes(colbytes), chan_bytes(chanbytes) {}

    int32_t dot(ImageVect coords) const;

    int32_t dot(int row, int col, int channel) const;
};

template <typename T>
class AddressCovector : public AddressCovectorBase {
  
  public:
    AddressCovector(const int16_t rowbytes, 
                    const int16_t colbytes, 
                    const int16_t chanbytes)
      : AddressCovectorBase(rowbytes, colbytes, chanbytes) {}

    AddressCovector(const unsigned width, 
                    const unsigned depth)
      : AddressCovectorBase(width*depth*sizeof(T), depth * sizeof(T), sizeof(T)) {}

    T* resolve(const T* base_address, const ImageVect& coords) const
      { return this->resolve(base_address, coords.row, coords.col, coords.channel); }

    T* resolve(const T* base_address, const int row, const int col, const int channel) const
      { return (T*)(((char*)base_address) + this->dot(row, col, channel)); }
};



}}