
#include "FilterGeometryIter.hpp"

#include <iostream>

using namespace nn::ff;

FilterGeometryIterator::FilterGeometryIterator(
    nn::Filter2dGeometry seed,
    std::initializer_list<IFilterFrame*> filter_frames)
    : frames(filter_frames), seed(seed) {
  frames.Push(seed);
}

FilterGeometryIterator::FilterGeometryIterator(
    std::initializer_list<IFilterFrame*> filter_frames)
    : FilterGeometryIterator(nn::Filter2dGeometry(), filter_frames) {}

void FilterGeometryIterator::Reset() { frames.Push(seed); }

FilterGeometryIterator& FilterGeometryIterator::operator++() {
  if (!frames.UpdateFilter()) frames = FrameStack();
  return *this;
}

bool FilterGeometryIterator::operator==(FilterGeometryIterator& other) const {
  // Two iterators are equal if the frame stack is equal
  return this->frames == other.frames;
}

bool FilterGeometryIterator::operator!=(FilterGeometryIterator& other) const {
  return this->frames != other.frames;
}

const nn::Filter2dGeometry& FilterGeometryIterator::operator*() {
  return this->frames.Filter();
}

FilterGeometryIterator& FilterGeometryIterator::begin() { return *this; }

static const FilterGeometryIterator END(nn::Filter2dGeometry(), {});

const FilterGeometryIterator& FilterGeometryIterator::end() { return END; }

FrameStack::FrameStack(std::initializer_list<IFilterFrame*> filter_frames) {
  for (IFilterFrame* frame : filter_frames)
    frames.push_back(std::shared_ptr<IFilterFrame>(frame));
}

const nn::Filter2dGeometry& FrameStack::Filter() {
  return this->frames.back()->Filter();
}

void FrameStack::Push(const nn::Filter2dGeometry& filter) {
  nn::Filter2dGeometry f = filter;
  for (int k = 0; k < frames.size(); k++) {
    frames[k]->Push(f);
    f = frames[k]->Filter();
  }
}

bool FrameStack::UpdateFilter() {
  for (int k = frames.size() - 1; k >= 0; --k) {
    if (frames[k]->UpdateFilter()) {
      while (k < (frames.size() - 1)) {
        frames[k + 1]->Push(frames[k]->Filter());
        k++;
      }

      return true;
    }
  }

  return false;
}

bool FrameStack::operator==(FrameStack& other) const {
  if (this->frames.size() != other.frames.size()) return false;
  for (int k = 0; k < this->frames.size(); k++)
    if (this->frames[k]->Filter() != other.frames[k]->Filter()) return false;

  return true;
}

bool FrameStack::operator!=(FrameStack& other) const {
  if (this->frames.size() != other.frames.size()) return true;
  for (int k = 0; k < this->frames.size(); k++)
    if (this->frames[k]->Filter() != other.frames[k]->Filter()) return true;

  return false;
}

FrameSequence::FrameSequence(std::initializer_list<IFilterFrame*> filter_frames)
    : current_index(0) {
  for (IFilterFrame* frame : filter_frames)
    frames.push_back(std::shared_ptr<IFilterFrame>(frame));
}

const nn::Filter2dGeometry& FrameSequence::Filter() {
  return Current()->Filter();
}

void FrameSequence::Push(const nn::Filter2dGeometry& filter) {
  current_index = 0;

  for (int k = 0; k < frames.size(); k++) {
    frames[k]->Push(filter);
  }
}

bool FrameSequence::UpdateFilter() {
  if (Current()->UpdateFilter()) return true;

  return (++current_index < frames.size());
}

Apply::Apply(BindFunc func) : func(func) {}

void Apply::Push(const nn::Filter2dGeometry& filter) {
  this->filter = func(filter);
}

AllUnpadded::AllUnpadded(const nn::Filter2dGeometry max_geometry,
                         const bool depthwise, const int channel_step)
    : max_geometry(max_geometry),
      channel_step(channel_step),
      depthwise(depthwise) {}

void AllUnpadded::Push(const nn::Filter2dGeometry& new_filter) {
  this->filter = nn::Filter2dGeometry(
      nn::ImageGeometry(1, 1, channel_step),
      nn::ImageGeometry(1, 1, channel_step),
      nn::WindowGeometry(1, 1, depthwise ? 1 : channel_step, 0, 0, 1, 1,
                         depthwise ? 1 : 0, 1, 1));
}

bool AllUnpadded::_UpdateFilter() {
#define UPDATE(FIELD, INIT, STEP)                        \
  do {                                                   \
    filter.FIELD += STEP;                                \
    if (filter.FIELD <= max_geometry.FIELD) return true; \
    filter.FIELD = INIT;                                 \
  } while (0)

  if (!depthwise) UPDATE(input.depth, channel_step, channel_step);

  UPDATE(output.depth, channel_step, channel_step);
  UPDATE(window.dilation.col, 1, 1);
  UPDATE(window.dilation.row, 1, 1);
  UPDATE(window.stride.col, 1, 1);
  UPDATE(window.stride.row, 1, 1);
  UPDATE(window.start.col, 0, 1);
  UPDATE(window.start.row, 0, 1);
  UPDATE(window.shape.width, 1, 1);
  UPDATE(window.shape.height, 1, 1);
  UPDATE(output.width, 1, 1);
  UPDATE(output.height, 1, 1);
#undef UPDATE

  return false;
}

bool AllUnpadded::UpdateFilter() {
  if (!_UpdateFilter()) return false;

  auto& input = filter.input;
  auto& output = filter.output;
  auto& window = filter.window;

  // Depthwise filters have input.depth == output.depth
  input.depth = depthwise ? output.depth : input.depth;
  // Depthwise filters have window.shape.depth == 1, dense filters have
  // window.shape.depth = input.depth
  window.shape.depth = depthwise ? 1 : input.depth;
  window.stride.channel = depthwise ? 1 : 0;

  // Now calculate the required input height and width

  input.height =
      window.start.row                           // start row for first output
      + (output.height - 1) * window.stride.row  // start row for last output
      + (window.shape.height - 1) *
            window.dilation.row  // last row for last output
      + 1;                       // input height

  input.width =
      window.start.col                          // start col for first output
      + (output.width - 1) * window.stride.col  // start col for last output
      + (window.shape.width - 1) *
            window.dilation.col  // last col for last output
      + 1;                       // input width

  return true;
}

AllPaddedBase::AllPaddedBase(const nn::Filter2dGeometry max_geometry,
                             const bool depthwise, const int channel_step)
    : max_geometry(max_geometry),
      channel_step(channel_step),
      depthwise(depthwise) {}

void AllPaddedBase::Push(const nn::Filter2dGeometry& new_filter) {
  this->filter = nn::Filter2dGeometry(
      nn::ImageGeometry(1, 1, channel_step),
      nn::ImageGeometry(1, 1, channel_step),
      nn::WindowGeometry(2, 2, depthwise ? 1 : channel_step, 0, 0, 1, 1,
                         depthwise ? 1 : 0, 1, 1));
}

bool AllPaddedBase::UpdateFilter() {
#define UPDATE(FIELD, INIT, STEP)                        \
  do {                                                   \
    filter.FIELD += STEP;                                \
    if (filter.FIELD <= max_geometry.FIELD) return true; \
    filter.FIELD = INIT;                                 \
  } while (0)

  if (!depthwise) UPDATE(input.depth, channel_step, channel_step);

  UPDATE(output.depth, channel_step, channel_step);
  UPDATE(window.dilation.col, 1, 1);
  UPDATE(window.dilation.row, 1, 1);
  UPDATE(window.stride.col, 1, 1);
  UPDATE(window.stride.row, 1, 1);
  UPDATE(window.shape.width, 2, 1);
  UPDATE(window.shape.height, 2, 1);
  UPDATE(output.width, 1, 1);
  UPDATE(output.height, 1, 1);

  return false;
#undef UPDATE
}

MakePadded::MakePadded(const nn::padding_t max_padding, const bool depthwise)
    : max_padding(max_padding), cur_padding{0, 0, 0, 0}, depthwise(depthwise) {}

void MakePadded::Push(const nn::Filter2dGeometry& new_filter) {
  // Padding order will be top, left, bottom, right, with right changing fastest
  this->filter = new_filter;
  this->cur_padding = {0, 0, 0, 0};
  UpdateFilter();
}

bool MakePadded::_UpdateFilter() {
  cur_padding.right++;
  if ((cur_padding.right <= max_padding.right) &&
      (cur_padding.right < filter.window.shape.width) &&
      (cur_padding.right + cur_padding.left < filter.input.width))
    return true;
  cur_padding.right = 0;

  cur_padding.bottom++;
  if ((cur_padding.bottom <= max_padding.bottom) &&
      (cur_padding.bottom < filter.window.shape.height) &&
      (cur_padding.bottom + cur_padding.top < filter.input.height))
    return true;
  cur_padding.bottom = 0;

  cur_padding.left++;
  if ((cur_padding.left <= max_padding.left) &&
      (cur_padding.left < filter.window.shape.width))
    return true;
  cur_padding.left = 0;

  cur_padding.top++;
  if ((cur_padding.top <= max_padding.top) &&
      (cur_padding.top < filter.window.shape.height))
    return true;

  return false;
}

bool MakePadded::UpdateFilter() {
  if (!_UpdateFilter()) return false;
  // std::cout << "Pad: " << cur_padding << std::endl;
  auto& input = filter.input;
  auto& output = filter.output;
  auto& window = filter.window;

  // Depthwise filters have input.depth == output.depth
  input.depth = depthwise ? output.depth : input.depth;
  // Depthwise filters have window.shape.depth == 1, dense filters have
  // window.shape.depth = input.depth
  window.shape.depth = depthwise ? 1 : input.depth;
  window.stride.channel = depthwise ? 1 : 0;

  // Compute input height/width for 0 padding
  input.height =
      (output.height - 1) * window.stride.row  // start row for last output
      + (window.shape.height - 1) *
            window.dilation.row  // last row for last output
      + 1;                       // input height

  input.width =
      (output.width - 1) * window.stride.col  // start col for last output
      + (window.shape.width - 1) *
            window.dilation.col  // last col for last output
      + 1;

  // Now adjust for padding
  input.height -= cur_padding.top + cur_padding.bottom;
  input.width -= cur_padding.left + cur_padding.right;

  window.start.row = -cur_padding.top;
  window.start.col = -cur_padding.left;

  return true;
}

IntDimensionRange::IntDimensionRange(int min, int max, int step)
    : min(min), max(max), step(step) {}

void IntDimensionRange::Push(const nn::Filter2dGeometry& new_filter) {
  this->filter = new_filter;
  Field() = min;
}

bool IntDimensionRange::UpdateFilter() {
  Field() += step;
  return Field() <= max;
}

nn::Filter2dGeometry nn::ff::MakeUnpaddedDepthwise(
    nn::Filter2dGeometry filter) {
  filter.window.start.row = 0;
  filter.window.start.col = 0;

  filter.window.stride.row = filter.window.shape.height;
  filter.window.stride.col = filter.window.shape.width;

  filter.window.dilation.row = 1;
  filter.window.dilation.col = 1;

  filter.input.height = filter.output.height * filter.window.shape.height;
  filter.input.width = filter.output.width * filter.window.shape.width;
  filter.input.depth = (filter.window.stride.channel == 1) ? filter.output.depth
                                                           : filter.input.depth;

  filter.window.shape.depth = (filter.window.stride.channel == 0)
                                  ? filter.input.depth
                                  : filter.window.shape.depth;
  return filter;
}

nn::Filter2dGeometry nn::ff::MakePaddedDepthwise(nn::Filter2dGeometry filter) {
  filter.window.start.row = 1 - filter.window.shape.height;
  filter.window.start.col = 1 - filter.window.shape.width;

  filter.window.stride.row = filter.window.shape.height;
  filter.window.stride.col = filter.window.shape.width;

  filter.window.dilation.row = 1;
  filter.window.dilation.col = 1;

  filter.input.height = (filter.output.height - 1) * filter.window.shape.height;
  filter.input.width = (filter.output.width - 1) * filter.window.shape.width;
  filter.input.depth = (filter.window.stride.channel == 1) ? filter.output.depth
                                                           : filter.input.depth;

  filter.window.shape.depth = (filter.window.stride.channel == 0)
                                  ? filter.input.depth
                                  : filter.window.shape.depth;
  return filter;
}

// nn::Filter2dGeometry nn::ff::MakeUnpaddedDepthwise(nn::Filter2dGeometry
// filter)
// {
//   filter.window.start.row = 0;
//   filter.window.start.col = 0;

//   filter.window.stride.row = filter.window.shape.height;
//   filter.window.stride.col = filter.window.shape.width;

//   filter.window.dilation.row = 1;
//   filter.window.dilation.col = 1;

//   filter.input.height = filter.output.height * filter.window.shape.height;
//   filter.input.width  = filter.output.width  * filter.window.shape.width;
//   filter.input.depth  = (filter.window.stride.channel == 1)?
//   filter.output.depth : filter.input.depth;

//   filter.window.shape.depth = (filter.window.stride.channel == 0)?
//   filter.input.depth : filter.window.shape.depth; return filter;
// }

// nn::Filter2dGeometry nn::ff::MakePaddedDepthwise(nn::Filter2dGeometry filter)
// {
//   filter.window.start.row = 1 - filter.window.shape.height;
//   filter.window.start.col = 1 - filter.window.shape.width;

//   filter.window.stride.row = filter.window.shape.height;
//   filter.window.stride.col = filter.window.shape.width;

//   filter.window.dilation.row = 1;
//   filter.window.dilation.col = 1;

//   filter.input.height = (filter.output.height-1) *
//   filter.window.shape.height; filter.input.width  = (filter.output.width -1)
//   * filter.window.shape.width; filter.input.depth  =
//   (filter.window.stride.channel == 1)? filter.output.depth :
//   filter.input.depth;

//   filter.window.shape.depth = (filter.window.stride.channel == 0)?
//   filter.input.depth : filter.window.shape.depth; return filter;
// }