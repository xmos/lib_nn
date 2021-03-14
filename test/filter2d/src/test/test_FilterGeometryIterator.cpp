

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/util/FilterGeometryIterator.hpp"
#include "xs3_vpu.h"

#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

#include <cstdint>
#include <memory>
#include <iostream>

using namespace nn::filt2d;


class MyFGI : public FilterGeometryIterator {

  protected:


  virtual void Next(FilterGeometry& filter) const override {
    filter = FilterGeometryIterator::END();
  }

};

static void printFilter(geom::Filter2dGeometry filter)
{

  std::cout << "Input Image:  { " << filter.input.height << ", " << filter.input.width << ", " 
            << filter.input.depth << " }" << std::endl;
  std::cout << "Output Image: { " << filter.output.height << ", " << filter.output.width << ", " 
            << filter.output.depth << " }" << std::endl;

  std::cout << "Window: " << "Shape { " << filter.window.shape.height << ", " 
                                        << filter.window.shape.width  << ", " 
                                        << filter.window.shape.depth << " }" << std::endl;
  std::cout << "        " << "Start { " << filter.window.start.row << ", "
                                        << filter.window.start.col << " }" << std::endl;
  std::cout << "        " << "Stride { " << filter.window.stride.row << ", "
                                         << filter.window.stride.col << ", "
                                         << filter.window.stride.channel << " }" << std::endl;
  std::cout << "        " << "Dilation { " << filter.window.dilation.row << ", "
                                           << filter.window.dilation.col << " }" << std::endl;

  std::cout << std::endl;


}

static int total = 1;
static int passed = 1;

bool MyPredicate(geom::Filter2dGeometry& filter)
{
  // printFilter(filter);
  total++;

  if(filter.ModelIsDepthwise()) {
    return false;
  }

  if(filter.ModelRequiresPadding()) {
    return false;
  }

  // Kind of unnecessary, because if it doesn't require padding, this should always
  // be true
  if(!filter.ModelConvWindowAlwaysIntersectsInput()) {
    return false;
  }

  passed++;
  return true;
}


TEST(FilterGeometryIterator,Blah){

  auto iter = PredicateFilterGeometryIterator(
                  geom::Filter2dGeometry(
                      geom::ImageGeometry(1, 1, VPU_INT8_EPV),
                      geom::ImageGeometry(1, 1, VPU_INT8_ACC_PERIOD),
                      geom::WindowGeometry(1, 1, VPU_INT8_EPV, 
                                                   0, 0,    1, 1, 0,    1, 1)),
                  geom::Filter2dGeometry(
                      geom::ImageGeometry(2, 2, VPU_INT8_EPV),
                      geom::ImageGeometry(2, 2, VPU_INT8_ACC_PERIOD),
                      geom::WindowGeometry(3, 3, VPU_INT8_EPV,
                                                   0, 0,    1, 1, 0,    1, 1)),
                  geom::Filter2dGeometry(
                      geom::ImageGeometry(1, 1, 0),
                      geom::ImageGeometry(1, 1, 0),
                      geom::WindowGeometry(1, 1, 0,
                                                   0, 0,    0, 0, 0,    0, 0)),
                  MyPredicate);

  for(auto filt: iter){

    std::cout << "A Filter:" << std::endl;
    printFilter(filt);

  }

  std::cout << passed << " / " << total << " geometries passed the predicate." << std::endl; 

};