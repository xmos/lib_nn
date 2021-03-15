
#include "FilterGeometryIterator.hpp"


using namespace nn::filt2d;

using FilterGeometry = geom::Filter2dGeometry;
using iterator = FilterGeometryIterator::iterator;


/////////////////////////
/////  FilterGeometryIterator
/////////////////////////

constexpr FilterGeometry FilterGeometryIterator::END;

iterator FilterGeometryIterator::begin() 
{
  return iterator(*this, this->First());
}


iterator FilterGeometryIterator::end() 
{
  return iterator(*this, FilterGeometryIterator::END);
}




/////////////////////////
/////  FilterGeometryIterator::iterator
/////////////////////////
iterator& FilterGeometryIterator::iterator::operator++()
{
  this->parent.Next(this->filter); 
  return *this;
}


iterator FilterGeometryIterator::iterator::operator++(int)
{
  iterator retval = *this; 
  ++(*this); 
  return retval;
}


bool FilterGeometryIterator::iterator::operator==(FilterGeometryIterator::iterator other) const
{
  return this->filter == other.filter;
}


bool FilterGeometryIterator::iterator::operator!=(FilterGeometryIterator::iterator other) const
{
  return this->filter != other.filter;
}


FilterGeometry& FilterGeometryIterator::iterator::operator*()
{
  return this->filter;
}




/////////////////////////
/////  PredicateFilterGeometryIterator
/////////////////////////

bool PredicateFilterGeometryIterator::InnerNext(
    FilterGeometry& filter) const
{
  #define UPDATE(FIELD)  do {                 \
    if(step.FIELD != 0) {                     \
      filter.FIELD += step.FIELD;             \
      if(filter.FIELD <= max.FIELD)           \
        return true;                          \
      filter.FIELD = min.FIELD; } } while(0)

  UPDATE(window.dilation.col);
  UPDATE(window.dilation.row);
  UPDATE(window.stride.channel);
  UPDATE(window.stride.col);
  UPDATE(window.stride.row);
  UPDATE(window.start.col);
  UPDATE(window.start.row);
  UPDATE(window.shape.depth);
  UPDATE(window.shape.width);
  UPDATE(window.shape.height);
  UPDATE(output.depth);
  UPDATE(output.width);
  UPDATE(output.height);
  UPDATE(input.depth);
  UPDATE(input.width);
  UPDATE(input.height);
  
  #undef UPDATE

  // If it made it here, we're done.
  return false;
}


FilterGeometry PredicateFilterGeometryIterator::First() const
{
  FilterGeometry filter = this->min;
  if(!predicate(filter))
    Next(filter);

  return filter;
}


void PredicateFilterGeometryIterator::Next(FilterGeometry& filter) const
{
  do {
    if(!InnerNext(filter)){
      filter = END;
      return;
    }
  } while(!predicate(filter));
}