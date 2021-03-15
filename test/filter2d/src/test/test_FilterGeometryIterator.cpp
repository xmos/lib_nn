

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/util/FilterGeometryIterator.hpp"
#include "xs3_vpu.h"
#include "../src/cpp/filt2d/Conv2dDeepFilter.hpp"

#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

#include <cstdint>
#include <memory>
#include <iostream>

using namespace nn::filt2d;


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


TEST(FilterGeometryIterator,Blah){



  int count = 0;
  for(auto filt: op::Conv2dDeepFilter_Valid::GetGeometryIterator()){
    printFilter(filt);
    count++;
  }

  std::cout << count << " geometries passed the predicate." << std::endl; 

};