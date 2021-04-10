#pragma once

#include "../src/cpp/filt2d/util/FilterGeometryIterator.hpp"
#include <functional>

namespace nn {
  namespace test {

class TestFilterGeometryIterator : public nn::FilterGeometryIterator {

  protected:
    FilterGeometry init;
    std::function<void(FilterGeometry&)> updater;
  public:

    TestFilterGeometryIterator(FilterGeometry init, 
                               std::function<void(FilterGeometry&)> updater)
        : init(init), updater(updater) {}
      
    virtual FilterGeometry First() const override { return init; }
    virtual void Next(FilterGeometry& filter) const override { updater(filter); }
}; 


}}