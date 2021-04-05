#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "Rand.hpp"
#include "../op/ref/ref_tests.hpp"

using namespace nn;



class Filter2dGeometryTest : public ::testing::TestWithParam<Filter2dGeometry> {};

// WindowGeometry is a really simple class. Mostly just a container for other values since I turned the 
// shape into an ImageGeometry

// TODO: I don't really like the way these tests are being done... I feel like I'm basically using the logic that
// the functions are using to test the functions

TEST_P(Filter2dGeometryTest, ModelIsDepthwise)
{
  auto filter = GetParam();

  ASSERT_EQ(filter.ModelIsDepthwise(), filter.window.stride.channel == 1);
}

TEST_P(Filter2dGeometryTest, ModelRequiresPadding)
{
  auto filter = GetParam();

  auto padded = false;
  padded = padded || filter.window.start.row < 0;
  padded = padded || filter.window.start.col < 0;
  padded = padded || (filter.window.start.row 
                      + filter.window.stride.row * (filter.output.height-1)
                      + filter.window.dilation.row * (filter.window.shape.height-1)) >= filter.input.height;
  padded = padded || (filter.window.start.col 
                      + filter.window.stride.col * (filter.output.width-1)
                      + filter.window.dilation.col * (filter.window.shape.width-1)) >= filter.input.width;

  ASSERT_EQ(filter.ModelRequiresPadding(), padded);
}


TEST_P(Filter2dGeometryTest, ModelConvWindowAlwaysIntersectsInput)
{
  auto filter = GetParam();

  auto pad_init = filter.ModelPadding(true, false);
  auto pad_fina = filter.ModelPadding(false, false);

  auto last_row_init = filter.window.start.row + (filter.window.shape.height - 1) * filter.window.dilation.row;
  auto last_col_init = filter.window.start.col + (filter.window.shape.width  - 1) * filter.window.dilation.col;

  auto first_row_final = filter.window.start.row + (filter.output.height - 1) * filter.window.stride.row
                         + (filter.window.shape.height - 1) * filter.window.dilation.row;
  auto first_col_final = filter.window.start.col + (filter.output.width  - 1) * filter.window.stride.col
                         + (filter.window.shape.width  - 1) * filter.window.dilation.col;

  auto top = (last_row_init >= 0);
  auto left = (last_col_init >= 0);
  auto bottom = (first_row_final < filter.input.height);
  auto right = (first_col_final < filter.input.width);

  auto val = filter.ModelConvWindowAlwaysIntersectsInput();

  auto always_intersects = top && left && bottom && right;

  std::cout << last_row_init << ", " << last_col_init << ", " << first_row_final << ", " << first_col_final << std::endl;

  ASSERT_EQ(val, always_intersects) << "(" << top << "," << left << "," << bottom << "," << right << ")";
}

// TEST_P(Filter2dGeometryTest, ModelConsumesInput)
// {
//   auto filter = GetParam();

//   auto consumes_input = true;

//   consumes_input = consumes_input && filter.window.start.row <= 0;
//   consumes_input = consumes_input && filter.window.start.col <= 0;

//   auto pad_final = filter.ModelPadding(false, true);

//   consumes_input = consumes_input && pad_final.right >= 0;
//   consumes_input = consumes_input && pad_final.bottom >= 0;

//   ASSERT_EQ(filter.ModelConsumesInput(), consumes_input);
// }

TEST_P(Filter2dGeometryTest, ModelPadding)
{
  auto filter = GetParam();

  auto pad_init_signed = filter.ModelPadding(true, true);
  auto pad_init_unsigned = filter.ModelPadding(true, false);

  ASSERT_EQ(pad_init_signed.top, -filter.window.start.row);
  ASSERT_EQ(pad_init_signed.left, -filter.window.start.col);
  ASSERT_EQ(pad_init_signed.bottom, (filter.window.start.row + filter.window.shape.height - 1) - filter.input.height);
  ASSERT_EQ(pad_init_signed.right, (filter.window.start.col + filter.window.shape.width - 1) - filter.input.width);

  ASSERT_EQ(pad_init_unsigned.top, std::max<int>(0, pad_init_signed.top));
  ASSERT_EQ(pad_init_unsigned.left, std::max<int>(0, pad_init_signed.left));
  ASSERT_EQ(pad_init_unsigned.bottom, std::max<int>(0, pad_init_signed.bottom));
  ASSERT_EQ(pad_init_unsigned.right, std::max<int>(0, pad_init_signed.right));

}

ParamedRandIter<Filter2dGeometry, SimpleFilter> iter[] = {
  ParamedRandIter<Filter2dGeometry, SimpleFilter>(200, SimpleFilter(false, false), 4356),
  ParamedRandIter<Filter2dGeometry, SimpleFilter>(200, SimpleFilter(false, true), 3456),
  ParamedRandIter<Filter2dGeometry, SimpleFilter>(200, SimpleFilter(true,  false, 16), 1211),
  ParamedRandIter<Filter2dGeometry, SimpleFilter>(200, SimpleFilter(true,  true,  31), 765),
};

INSTANTIATE_TEST_SUITE_P(Simple, Filter2dGeometryTest, ::testing::Values<Filter2dGeometry>(
    Filter2dGeometry( ImageGeometry( 1, 1, 1), ImageGeometry( 1, 1, 1), WindowGeometry( 1, 1, 1,   0, 0,   1, 1, 1,   1, 1)),
    Filter2dGeometry( ImageGeometry( 2, 2, 1), ImageGeometry( 1, 1, 1), WindowGeometry( 1, 1, 1,   0, 0,   1, 1, 1,   1, 1)),
    Filter2dGeometry( ImageGeometry( 2, 2, 1), ImageGeometry( 2, 2, 1), WindowGeometry( 1, 1, 1,   0, 0,   1, 1, 1,   1, 1)), 
    Filter2dGeometry( ImageGeometry( 1, 1, 1), ImageGeometry( 2, 2, 1), WindowGeometry( 2, 2, 1,   0, 0,   1, 1, 1,   1, 1)), 
    Filter2dGeometry( ImageGeometry( 5,10, 1), ImageGeometry( 3, 4, 1), WindowGeometry( 3, 4, 1,  -2,-3,   3, 4, 0,   1, 1))
));

INSTANTIATE_TEST_SUITE_P(DenseNoPad,      Filter2dGeometryTest, ::testing::ValuesIn(iter[0].begin(), iter[0].end())); 
INSTANTIATE_TEST_SUITE_P(DensePadded,     Filter2dGeometryTest, ::testing::ValuesIn(iter[1].begin(), iter[1].end())); 
INSTANTIATE_TEST_SUITE_P(DepthwiseNoPad,  Filter2dGeometryTest, ::testing::ValuesIn(iter[2].begin(), iter[2].end())); 
INSTANTIATE_TEST_SUITE_P(DepthwisePadded, Filter2dGeometryTest, ::testing::ValuesIn(iter[3].begin(), iter[3].end())); 

