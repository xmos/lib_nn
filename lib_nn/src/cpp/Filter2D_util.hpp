#pragma once

#include "f2d_c_types.h"
#include <type_traits>

namespace nn {
namespace filt2d {


class ImageVect {

  public:

    const int row;
    const int col;
    const int channel;

    ImageVect(
      int const img_row,
      int const img_col,
      int const img_chan)
        : row(img_row), col(img_col), channel(img_chan){}

    ImageVect operator+(ImageVect const& other) const
      { return this->add(other.row, other.col, other.channel);  }

    ImageVect operator-(ImageVect const& other) const
      { return this->sub(other.row, other.col, other.channel);  }

    ImageVect add(int const rows, int const cols, int const chans) const
      { return ImageVect(this->row + rows, this->col + cols, this->channel + chans); }

    ImageVect sub(int const rows, int const cols, int const chans) const
      { return ImageVect(this->row - rows, this->col - cols, this->channel - chans); }
};


class ImageRegion {

  public:

    struct {
      const int row;
      const int col;
      const int channel;
    } start;

    struct {
      const unsigned height;
      const unsigned width;
      const unsigned depth;
    } shape;

  public:

    ImageRegion(
      int const row,
      int const col,
      int const chan,
      unsigned const height,
      unsigned const width,
      unsigned const depth)
        : start{row,col,chan}, shape{height,width,depth} {}

};


////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

template <typename T>
class IMemCopyHandler {

  IMemCopyHandler() = delete;

  public:

    T const* copy_mem(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};

template <typename T_elm_in, typename T_acc>
class IAggregationHandler {

  IAggregationHandler() = delete;

  public:

    T_acc aggregate(
      T_elm_in const* input_img,
      ImageVect const& output_coords,
      unsigned const channels_out);

};

template <typename T_acc, typename T_out>
class IOutputTransformHandler {

  IOutputTransformHandler() = delete;

  public:

    void transform(
      T_out * output,
      T_acc const& accumulator,
      unsigned const channels_out);

};


////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

class ICoordinateConverter {

  ICoordinateConverter() = delete;

  protected:

    InputCoordTransform const& getTransform() const;
    ImageVect getInputCoords(ImageVect const& output_coords) const;

};


class IPaddingResolver {
  IPaddingResolver() = delete;
  protected:
    PaddingTransform const& getPaddingTransform() const;
    padding_t const getPadding(
      ImageVect const& output_coords,
      bool const get_unsigned) const;
};

}}