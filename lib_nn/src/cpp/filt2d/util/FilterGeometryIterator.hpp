#pragma once

#include "../geom/Filter2dGeometry.hpp"

#include <iostream>
#include <functional>

namespace nn {

class FilterGeometryIterator {

  public:

    using FilterGeometry = Filter2dGeometry;

    static constexpr FilterGeometry END = FilterGeometry(ImageGeometry(0,0,0), 
                                            ImageGeometry(0,0,0),
                                            WindowGeometry(0,0,0,0,0,0,0,0,0,0));

    class iterator: public std::iterator<std::input_iterator_tag, FilterGeometry> {
      private:
        FilterGeometryIterator& parent;
        FilterGeometry filter;
      public:
        explicit iterator(FilterGeometryIterator& parent, FilterGeometry filter)
                      : parent(parent), filter(filter) {}
        iterator& operator++();
        iterator operator++(int);
        bool operator==(iterator other) const;
        bool operator!=(iterator other) const;
        reference operator*();
    };

  protected:
    virtual FilterGeometry First() const = 0;
    virtual void Next(FilterGeometry& filter) const = 0;
  public:
    iterator begin();
    iterator end();
};



class PredicateFilterGeometryIterator : public FilterGeometryIterator {

  public:
    using FilterGeometryPredicate = std::function<bool(FilterGeometry&)>;

  protected:
    FilterGeometry min;
    FilterGeometry max;
    FilterGeometry step;
    FilterGeometryPredicate predicate;
    
    virtual bool InnerNext(FilterGeometry& filter) const;

  public:

    PredicateFilterGeometryIterator(
        const FilterGeometry min, 
        const FilterGeometry max, 
        const FilterGeometry step,
        FilterGeometryPredicate predicate) noexcept
          : min(min), max(max), step(step), predicate(predicate) {}
      
    virtual FilterGeometry First() const override;
    virtual void Next(FilterGeometry& filter) const override;
}; 



}