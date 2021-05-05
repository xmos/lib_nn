#pragma once

#include "nn_api.h"

#include <iostream>
#include <array>



#define CONV2D_OUTPUT_LENGTH(input_length, filter_size, dilation, stride)     \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)



namespace nn {
  

  C_API typedef struct {
    int16_t top;
    int16_t left;
    int16_t bottom;
    int16_t right;

    void MakeUnsigned(){
      top = std::max<int16_t>(0, top);
      left = std::max<int16_t>(0, left);
      bottom = std::max<int16_t>(0, bottom);
      right = std::max<int16_t>(0, right);
    }

    bool HasPadding() const {
      return top > 0 || left > 0 || bottom > 0 || right > 0;
    }
  } padding_t;



  class ImageVect {

   
    public:

      int row;
      int col;
      int channel;

      ImageVect(
        int const img_row,
        int const img_col,
        int const img_chan)
          : row(img_row), col(img_col), channel(img_chan){}

      ImageVect(const std::array<int,3> coords)
          : ImageVect(coords[0], coords[1], coords[2]) {}

      ImageVect operator+(ImageVect const& other) const
        { return this->add(other.row, other.col, other.channel);  }

      ImageVect operator-(ImageVect const& other) const
        { return this->sub(other.row, other.col, other.channel);  }

      ImageVect add(int const rows, int const cols, int const chans) const
        { return ImageVect(this->row + rows, this->col + cols, this->channel + chans); }

      ImageVect sub(int const rows, int const cols, int const chans) const
        { return ImageVect(this->row - rows, this->col - cols, this->channel - chans); }

      bool operator==(const ImageVect& other) const 
        { return (row==other.row)&&(col==other.col)&&(channel==other.channel); }
      bool operator!=(const ImageVect& other) const 
        { return !((row==other.row)&&(col==other.col)&&(channel==other.channel)); }
  };


  class ImageRegion {

    public:

      struct {
        const int row;
        const int col;
        const int channel;
      } start;

      struct {
        const int height;
        const int width;
        const int depth;
      } shape;

    public:

      ImageRegion(
        int const row,
        int const col,
        int const chan,
        int const height,
        int const width,
        int const depth)
          : start{row,col,chan}, shape{height,width,depth} {}

      ImageVect startVect() const
        { return ImageVect(start.row, start.col, start.channel); }
      ImageVect endVect(bool inclusive = false) const 
        { return ImageVect(start.row + shape.height + (inclusive? -1 : 0), 
                           start.col + shape.width + (inclusive? -1 : 0), 
                           start.channel + shape.depth + (inclusive? -1 : 0)); }

      bool Within(int row, int col, int channel) const
      {
        if( row < start.row || row >= (start.row + shape.height) ) return false;
        if( col < start.col || col >= (start.col + shape.width) ) return false;
        if( channel < start.channel || channel >= (start.channel + shape.depth) ) return false;
        return true;
      }

      int PixelCount() const
      { return shape.height * shape.width; }

      int ElementCount() const
      { return PixelCount() * shape.depth; }

      int ChannelOutputGroups(int output_channels_per_group) const
      { return (shape.depth + (output_channels_per_group - 1)) / output_channels_per_group; }

  };


  inline std::ostream& operator<<(std::ostream &stream, const padding_t &pad){
    return stream << "(" << pad.top << "," << pad.left << "," << pad.bottom << "," << pad.right << ")";
  }

  inline std::ostream& operator<<(std::ostream &stream, const ImageVect &vect){
    return stream << "(" << vect.row << "," << vect.col << "," << vect.channel << ")";
  }


  inline std::ostream& operator<<(std::ostream &stream, const ImageRegion &r){
    const auto end = r.endVect();
    return stream << "{ [" << r.start.row << "," << end.row << "), "
                  <<   "[" << r.start.col << "," << end.col << "), "
                  <<   "[" << r.start.channel << "," << end.channel << ") }";
  }
}