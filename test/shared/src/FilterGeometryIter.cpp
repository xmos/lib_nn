
#include <iostream>

#include "FilterGeometryIter.hpp"



using namespace nn::ff;


FilterGeometryIterator::FilterGeometryIterator(nn::Filter2dGeometry seed,
                                               std::initializer_list<IFilterFrame*> filter_frames)
    : frames( filter_frames )
{
  frames.Push( seed );
}

FilterGeometryIterator& FilterGeometryIterator::operator++()
{
  if (!frames.UpdateFilter())
    frames = FrameStack();
  return *this;
}

bool FilterGeometryIterator::operator==(FilterGeometryIterator& other) const
{
  // Two iterators are equal if the frame stack is equal
  return this->frames == other.frames;
}

bool FilterGeometryIterator::operator!=(FilterGeometryIterator& other) const
{
  return this->frames != other.frames;
}

const nn::Filter2dGeometry& FilterGeometryIterator::operator*()
{
  return this->frames.Filter();
}


FilterGeometryIterator& FilterGeometryIterator::begin()
{
  return *this;
}

static const FilterGeometryIterator END(nn::Filter2dGeometry(), {});

const FilterGeometryIterator& FilterGeometryIterator::end()
{
  return END;
}






FrameStack::FrameStack(std::initializer_list<IFilterFrame*> filter_frames)
{
    for(IFilterFrame* frame: filter_frames)
      frames.push_back(std::shared_ptr<IFilterFrame>(frame));
}

const nn::Filter2dGeometry& FrameStack::Filter()
{
  return this->frames.back()->Filter();
}

void FrameStack::Push(const nn::Filter2dGeometry& filter)
{
  nn::Filter2dGeometry f = filter;
  for(int k = 0; k < frames.size(); k++){
    frames[k]->Push( f );
    f = frames[k]->Filter();
  }
}

bool FrameStack::UpdateFilter()
{
  for(int k = frames.size()-1; k >= 0; --k) {
    if( frames[k]->UpdateFilter() ) {

      while( k < (frames.size() - 1) ){
        frames[k+1]->Push( frames[k]->Filter() );
        k++;
      }

      return true;
    }
  }

  return false;
}

bool FrameStack::operator==(FrameStack& other) const
{
  if(this->frames.size() != other.frames.size()) return false;
  for(int k = 0; k < this->frames.size(); k++)
    if(this->frames[k]->Filter() != other.frames[k]->Filter()) return false;
  
  return true;
}

bool FrameStack::operator!=(FrameStack& other) const
{
  if(this->frames.size() != other.frames.size()) return true;
  for(int k = 0; k < this->frames.size(); k++)
    if(this->frames[k]->Filter() != other.frames[k]->Filter()) return true;

  return false;
}










FrameSequence::FrameSequence(std::initializer_list<IFilterFrame*> filter_frames)
  : current_index(0)
{
    for(IFilterFrame* frame: filter_frames)
      frames.push_back(std::shared_ptr<IFilterFrame>(frame));
}

const nn::Filter2dGeometry& FrameSequence::Filter()
{
  return Current()->Filter();
}

void FrameSequence::Push(const nn::Filter2dGeometry& filter)
{
  current_index = 0;

  for(int k = 0; k < frames.size(); k++){
    frames[k]->Push( filter );
  }
} 

bool FrameSequence::UpdateFilter()
{
  if( Current()->UpdateFilter() )
    return true;
  
  return (++current_index < frames.size());
}




Apply::Apply( BindFunc func ) : func(func) { }

void Apply::Push(const nn::Filter2dGeometry& filter)
{
  this->filter = func(filter);
}




IntDimensionRange::IntDimensionRange(int min, int max, int step) : min(min), max(max), step(step) {}

void IntDimensionRange::Push(const nn::Filter2dGeometry& new_filter)
{
  this->filter = new_filter;
  Field() = min;
}

bool IntDimensionRange::UpdateFilter()
{
  Field() += step;
  return Field() <= max;
}



nn::Filter2dGeometry nn::ff::MakeUnpadded(nn::Filter2dGeometry filter)
{
  filter.window.start.row = 0;
  filter.window.start.col = 0;

  filter.window.stride.row = filter.window.shape.height;
  filter.window.stride.col = filter.window.shape.width;
  
  filter.window.dilation.row = 1;
  filter.window.dilation.col = 1;

  filter.input.height = filter.output.height * filter.window.shape.height;
  filter.input.width  = filter.output.width  * filter.window.shape.width;
  filter.input.depth  = (filter.window.stride.channel == 1)? filter.output.depth : filter.input.depth;

  filter.window.shape.depth = (filter.window.stride.channel == 0)? filter.input.depth : filter.window.shape.depth;
  return filter;
}


nn::Filter2dGeometry nn::ff::MakePadded(nn::Filter2dGeometry filter)
{
  filter.window.start.row = 1 - filter.window.shape.height;
  filter.window.start.col = 1 - filter.window.shape.width;

  filter.window.stride.row = filter.window.shape.height;
  filter.window.stride.col = filter.window.shape.width;

  filter.window.dilation.row = 1;
  filter.window.dilation.col = 1;

  filter.input.height = (filter.output.height-1) * filter.window.shape.height;
  filter.input.width  = (filter.output.width -1) * filter.window.shape.width;
  filter.input.depth  = (filter.window.stride.channel == 1)? filter.output.depth : filter.input.depth;

  filter.window.shape.depth = (filter.window.stride.channel == 0)? filter.input.depth : filter.window.shape.depth;
  return filter;
}


