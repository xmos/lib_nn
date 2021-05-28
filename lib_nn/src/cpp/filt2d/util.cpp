#include "geom/util.hpp"

using namespace nn;

/////////////////////////////////////////////////////////////////
//                    padding_t
/////////////////////////////////////////////////////////////////
void padding_t::MakeUnsigned() {
  top = std::max<int16_t>(0, top);
  left = std::max<int16_t>(0, left);
  bottom = std::max<int16_t>(0, bottom);
  right = std::max<int16_t>(0, right);
}

bool padding_t::HasPadding() const {
  return top > 0 || left > 0 || bottom > 0 || right > 0;
}

bool padding_t::operator==(const padding_t &other) const {
  return (top == other.top) && (left == other.left) &&
         (bottom == other.bottom) && (right == other.right);
}

bool padding_t::operator!=(const padding_t &other) const {
  return (top != other.top) || (left != other.left) ||
         (bottom != other.bottom) || (right != other.right);
}

/////////////////////////////////////////////////////////////////
//                    ImageVect
/////////////////////////////////////////////////////////////////
ImageVect ImageVect::operator+(const ImageVect &other) const {
  return this->add(other.row, other.col, other.channel);
}

ImageVect ImageVect::operator-(ImageVect const &other) const {
  return this->sub(other.row, other.col, other.channel);
}

ImageVect ImageVect::add(int const rows, int const cols,
                         int const chans) const {
  return ImageVect(this->row + rows, this->col + cols, this->channel + chans);
}

ImageVect ImageVect::sub(int const rows, int const cols,
                         int const chans) const {
  return ImageVect(this->row - rows, this->col - cols, this->channel - chans);
}

bool ImageVect::operator==(const ImageVect &other) const {
  return (row == other.row) && (col == other.col) && (channel == other.channel);
}
bool ImageVect::operator!=(const ImageVect &other) const {
  return !((row == other.row) && (col == other.col) &&
           (channel == other.channel));
}

/////////////////////////////////////////////////////////////////
//                    ImageRegion
/////////////////////////////////////////////////////////////////
ImageVect ImageRegion::StartVect() const {
  return ImageVect(start.row, start.col, start.channel);
}

ImageVect ImageRegion::EndVect(bool inclusive) const {
  return ImageVect(start.row + shape.height + (inclusive ? -1 : 0),
                   start.col + shape.width + (inclusive ? -1 : 0),
                   start.channel + shape.depth + (inclusive ? -1 : 0));
}

bool ImageRegion::Within(int row, int col, int channel) const {
  if (row < start.row || row >= (start.row + shape.height)) return false;
  if (col < start.col || col >= (start.col + shape.width)) return false;
  if (channel < start.channel || channel >= (start.channel + shape.depth))
    return false;
  return true;
}

int ImageRegion::PixelCount() const { return shape.height * shape.width; }

int ImageRegion::ElementCount() const { return PixelCount() * shape.depth; }

int ImageRegion::ChannelOutputGroups(int output_channels_per_group) const {
  return (shape.depth + (output_channels_per_group - 1)) /
         output_channels_per_group;
}
