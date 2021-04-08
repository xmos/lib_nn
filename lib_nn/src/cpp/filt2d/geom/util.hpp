#pragma once

#include <iostream>

namespace nn {

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