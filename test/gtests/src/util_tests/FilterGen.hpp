#pragma once

#include <vector>

#include "geom/Filter2dGeometry.hpp"

namespace nn {
namespace test {
namespace filt_gen {

/**
 * Generate a vector of simple Filter2dGeometry object for testing.
 *
 * In this case "simple" means that no padding is required, no dilation is used,
 * and the filter dimensions are equal to its strides.
 *
 * `Y_dim` and `K_dim` are the maximum spacial dimensions of the output image
 * and filter window respectively. The input image dimensions will be calculated
 * from these.
 *
 * A total of (Y_dim^2)*(K_dim^2) geometries will be output.
 */
std::vector<Filter2dGeometry> SimpleFilters(int Y_dim = 3, int K_dim = 3) {
  auto res = std::vector<Filter2dGeometry>();

  for (int K_h = 1; K_h <= K_dim; K_h++) {
    for (int K_w = 1; K_w <= K_dim; K_w++) {
      WindowGeometry win(K_h, K_w, 1, 0, 0, K_h, K_w, 0, 1, 1);

      for (int Y_h = 1; Y_h <= Y_dim; Y_h++) {
        for (int Y_w = 1; Y_w <= Y_dim; Y_w++) {
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

/**
 * Generate a vector of Filter2dGeometry objects that require padding for
 * testing.
 */
std::vector<Filter2dGeometry> PaddedFilters(int Y_dim = 3, int K_dim = 3) {
  auto res = std::vector<Filter2dGeometry>();

  for (int K_h = 1; K_h <= K_dim; K_h++) {
    for (int K_w = 1; K_w <= K_dim; K_w++) {
      if (K_h == 1 && K_w == 1) continue;

      WindowGeometry win(K_h, K_w, 1, 1 - K_h, 1 - K_w, K_h, K_w, 0, 1, 1);

      for (int Y_h = 1; Y_h <= Y_dim; Y_h++) {
        for (int Y_w = 1; Y_w <= Y_dim; Y_w++) {
          int last_start_row = win.start.row + (Y_h - 1) * win.stride.row;
          int last_start_col = win.start.col + (Y_w - 1) * win.stride.col;

          int X_h = (Y_h == 1) ? 1 : (last_start_row + 1);
          int X_w = (Y_w == 1) ? 1 : (last_start_col + 1);

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

/**
 * Generate a vector of Filter2dGeometry objects that use dilation, but don't
 * require any padding
 */
std::vector<Filter2dGeometry> DilatedFilters(int Y_dim = 3, int K_dim = 3) {
  auto res = std::vector<Filter2dGeometry>();

  for (int K_h = 2; K_h <= K_dim; K_h++) {
    for (int K_w = 2; K_w <= K_dim; K_w++) {
      if (K_h == 1 && K_w == 1) continue;

      for (int D_h = 2; D_h <= 3; D_h++) {
        for (int D_w = 2; D_w <= 3; D_w++) {
          WindowGeometry win(K_h, K_w, 1, 0, 0, K_h, K_w, 0, D_h, D_w);

          for (int Y_h = 1; Y_h <= Y_dim; Y_h++) {
            for (int Y_w = 1; Y_w <= Y_dim; Y_w++) {
              int last_start_row = win.start.row + (Y_h - 1) * win.stride.row +
                                   (K_h - 1) * win.dilation.row;
              int last_start_col = win.start.col + (Y_w - 1) * win.stride.col +
                                   (K_w - 1) * win.dilation.col;

              int X_h = last_start_row + 1;
              int X_w = last_start_col + 1;

              ImageGeometry output(Y_h, Y_w, 1);
              ImageGeometry input(X_h, X_w, 1);

              auto filt = Filter2dGeometry(input, output, win);
              res.push_back(filt);
            }
          }
        }
      }
    }
  }

  return res;
}

}  // namespace filt_gen
}  // namespace test
}  // namespace nn