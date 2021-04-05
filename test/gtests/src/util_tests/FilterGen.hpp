#pragma once

#include <vector>

#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"

namespace nn {
  namespace test {
    namespace filt_gen {

/**
 * Generate a vector of simple Filter2dGeometry object for testing.
 * 
 * In this case "simple" means that no padding is required, no dilation is used, and the filter dimensions
 * are equal to its strides.
 * 
 * `Y_dim` and `K_dim` are the maximum spacial dimensions of the output image and filter window respectively.
 * The input image dimensions will be calculated from these.
 * 
 * A total of (Y_dim^2)*(K_dim^2) geometries will be output.
 */
std::vector<Filter2dGeometry>
SimpleFilters(int Y_dim = 3, int K_dim = 3)
{

  auto res = std::vector<Filter2dGeometry>();

  for(int K_h = 1; K_h <= K_dim; K_h++){
    for(int K_w = 1; K_w <= K_dim; K_w++){

      WindowGeometry win(K_h, K_w, 1,  0, 0,  K_h, K_w,   0, 1, 1);

      for(int Y_h = 1; Y_h <= Y_dim; Y_h++){
        for(int Y_w = 1; Y_w <= Y_dim; Y_w++){

          int X_h = Y_h * K_h;
          int X_w = Y_w * K_w;

          ImageGeometry output(Y_h, Y_w, 1);
          ImageGeometry input(X_h, X_w, 1);

          auto filt = Filter2dGeometry(input, output, win);
          res.push_back(filt);

        }
      }

    }
  }

  return res;
}





}}}