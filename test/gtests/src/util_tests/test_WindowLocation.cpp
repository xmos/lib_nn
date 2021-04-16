
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "../src/cpp/filt2d/geom/WindowLocation.hpp"
#include "Rand.hpp"
#include "FilterGen.hpp"

using namespace nn;



class WindowLocationTest : public ::testing::TestWithParam<Filter2dGeometry> {};



TEST_P(WindowLocationTest, InputStart)
{

  auto filter = GetParam();

  ImageVect exp(0,0,0);

  exp.row = filter.window.start.row;
  for(int yr = 0; yr < filter.output.height; yr++){

    exp.col = filter.window.start.col;
    for(int yc = 0; yc < filter.output.width; yc++){

      exp.channel = 0;
      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        auto start = loc.InputStart();

        EXPECT_EQ(start.row, exp.row) << "Output Coords: " << loc.output_coords;
        EXPECT_EQ(start.col, exp.col) << "Output Coords: " << loc.output_coords;
        EXPECT_EQ(start.channel, exp.channel) << "Output Coords: " << loc.output_coords;

        exp.channel += filter.window.stride.channel;
      }

      exp.col += filter.window.stride.col;
    }

    exp.row += filter.window.stride.row;
  }
}


TEST_P(WindowLocationTest, InputEnd)
{

  auto filter = GetParam();

  ImageVect exp(0,0,0);

  exp.row = filter.window.start.row + (filter.window.shape.height - 1) * filter.window.dilation.row;
  for(int yr = 0; yr < filter.output.height; yr++){

    exp.col = filter.window.start.col + (filter.window.shape.width - 1) * filter.window.dilation.col;
    for(int yc = 0; yc < filter.output.width; yc++){

      exp.channel = filter.window.shape.depth - 1;
      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        auto end = loc.InputEnd();

        EXPECT_EQ(end.row, exp.row)         << "Output Coords: " << loc.output_coords;
        EXPECT_EQ(end.col, exp.col)         << "Output Coords: " << loc.output_coords;
        EXPECT_EQ(end.channel, exp.channel) << "Output Coords: " << loc.output_coords;

        exp.channel += filter.window.stride.channel;
      }

      exp.col += filter.window.stride.col;
    }

    exp.row += filter.window.stride.row;
  }
}



TEST_P(WindowLocationTest, InputCoords)
{

  auto filter = GetParam();

  ImageVect exp(0,0,0);

  exp.row = filter.window.start.row + (filter.window.shape.height - 1) * filter.window.dilation.row;
  for(int yr = 0; yr < filter.output.height; yr++){

    exp.col = filter.window.start.col + (filter.window.shape.width - 1) * filter.window.dilation.col;
    for(int yc = 0; yc < filter.output.width; yc++){

      exp.channel = filter.window.shape.depth - 1;
      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        auto start = loc.InputStart();
        auto exp = start;

        for(int kr = 0; kr < filter.window.shape.height; kr++){

          exp.col = start.col;
          for(int kc = 0; kc < filter.window.shape.width; kc++){

            exp.channel = start.channel;
            for(int kx = 0; kx < filter.window.shape.depth; kx++){

              auto in_coords = loc.InputCoords(kr, kc, kx);

              EXPECT_EQ(in_coords.row, exp.row)         << "Output Coords: " << loc.output_coords << "| Filter Coords: " << ImageVect(kr, kc, kx);
              EXPECT_EQ(in_coords.col, exp.col)         << "Output Coords: " << loc.output_coords << "| Filter Coords: " << ImageVect(kr, kc, kx);
              EXPECT_EQ(in_coords.channel, exp.channel) << "Output Coords: " << loc.output_coords << "| Filter Coords: " << ImageVect(kr, kc, kx);

              exp.channel += 1;
            }

            exp.col += filter.window.dilation.col;
          }

          exp.row += filter.window.dilation.row;
        }

        exp.channel += filter.window.stride.channel;
      }

      exp.col += filter.window.stride.col;
    }

    exp.row += filter.window.stride.row;
  }
}


TEST_P(WindowLocationTest, Padding)
{

  auto filter = GetParam();

  auto pad_start = filter.ModelPadding(true, true);

  auto exp_pad = pad_start;

  for(int yr = 0; yr < filter.output.height; yr++){

    exp_pad.left = pad_start.left;
    exp_pad.right = pad_start.right;
    for(int yc = 0; yc < filter.output.width; yc++){

      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        auto actual = loc.Padding();

        EXPECT_EQ( std::max<int>( 0, exp_pad.top    ), actual.top    ) << "Y: " << ImageVect(yr,yc,yx);
        EXPECT_EQ( std::max<int>( 0, exp_pad.left   ), actual.left   ) << "Y: " << ImageVect(yr,yc,yx);
        EXPECT_EQ( std::max<int>( 0, exp_pad.bottom ), actual.bottom ) << "Y: " << ImageVect(yr,yc,yx);
        EXPECT_EQ( std::max<int>( 0, exp_pad.right  ), actual.right  ) << "Y: " << ImageVect(yr,yc,yx);

      }

      exp_pad.left  -= filter.window.stride.col;
      exp_pad.right += filter.window.stride.col;
    }

    exp_pad.top    -= filter.window.stride.row;
    exp_pad.bottom += filter.window.stride.row;
  }
}


TEST_P(WindowLocationTest, IsPadding)
{
  auto rand = nn::test::Rand(4564523);

  auto filter = GetParam();

  auto input_img = std::vector<int8_t>( filter.input.imageElements() );

  for(int k = 0; k < input_img.size(); k++){
    input_img[k] = rand.rand<int8_t>();
  }

  for(int yr = 0; yr < filter.output.height; yr++){

    for(int yc = 0; yc < filter.output.width; yc++){

      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        for(int kr = 0; kr < filter.window.shape.height; kr++){
          for(int kc = 0; kc < filter.window.shape.width; kc++){
            for(int kx = 0; kx < filter.window.shape.depth; kx++){

              auto in_coords = loc.InputCoords(kr, kc, kx);

              auto expected = !filter.input.IsWithinImage(in_coords);
              auto actual = loc.IsPadding(kr, kc, kx); 

              EXPECT_EQ(expected, actual) << "Output Coords: " << loc.output_coords 
                                          << "| Filter Coords: " << ImageVect(kr, kc, kx);
            }
          }
        }
      }
    }
  }
}



TEST_P(WindowLocationTest, InputElement)
{
  auto filter = GetParam();

  auto input_img = std::vector<int8_t>( filter.input.imageElements() );

  for(int yr = 0; yr < filter.output.height; yr++){

    for(int yc = 0; yc < filter.output.width; yc++){

      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        for(int kr = 0; kr < filter.window.shape.height; kr++){
          
          if( loc.IsPadding(kr, 0) )
            continue;

          for(int kc = 0; kc < filter.window.shape.width; kc++){

            if( loc.IsPadding(kr, kc) )
              continue;

            for(int kx = 0; kx < filter.window.shape.depth; kx++){

              auto in_coords = loc.InputCoords(kr, kc, kx);

              int index = (in_coords.row * int(filter.input.depth * filter.input.width))
                         +(in_coords.col * int(filter.input.depth))
                         +(in_coords.channel);

              int8_t* expected = &input_img[index];
              int8_t* actual = &loc.InputElement(&input_img[0], kr, kc, kx);

              EXPECT_EQ(expected, actual) << "Output Coords: " << loc.output_coords 
                                          << " | Filter Coords: " << ImageVect(kr, kc, kx)
                                          << " | in_coords: " << in_coords
                                          << " | index: " << index
                                          << " | input_img: " << &input_img[0];
            }
          }
        }
      }
    }
  }
}



TEST_P(WindowLocationTest, GetInput)
{
  auto rand = nn::test::Rand(754444);

  auto filter = GetParam();

  auto input_img = std::vector<int8_t>( filter.input.imageElements() );

  for(int k = 0; k < input_img.size(); k++){
    input_img[k] = rand.rand<int8_t>();
  }

  auto zero_pad = rand.rand<int8_t>();

  for(int yr = 0; yr < filter.output.height; yr++){

    for(int yc = 0; yc < filter.output.width; yc++){

      for(int yx = 0; yx < filter.output.depth; yx++){
        auto loc = WindowLocation(filter, ImageVect(yr,yc,yx));

        for(int kr = 0; kr < filter.window.shape.height; kr++){
          for(int kc = 0; kc < filter.window.shape.width; kc++){
            for(int kx = 0; kx < filter.window.shape.depth; kx++){

              auto in_coords = loc.InputCoords(kr, kc, kx);

              int index = (in_coords.row * int(filter.input.depth * filter.input.width))
                         +(in_coords.col * int(filter.input.depth))
                         +(in_coords.channel);

              int8_t expected = loc.IsPadding(kr,kc,kx)? zero_pad : input_img[index];
              int8_t actual = loc.GetInput<int8_t>(&input_img[0], kr, kc, kx, zero_pad);

              ASSERT_EQ(expected, actual) << "Output Coords: " << loc.output_coords 
                                          << " | Filter Coords: " << ImageVect(kr, kc, kx)
                                          << " | in_coords: " << in_coords
                                          << " | index: " << index
                                          << " | input_img: " << &input_img[0]
                                          << " | zero_pad: " << int(zero_pad);
            }
          }
        }
      }
    }
  }
}


static auto simple_filters = ::testing::ValuesIn( nn::test::filt_gen::SimpleFilters() );
static auto padded_filters = ::testing::ValuesIn( nn::test::filt_gen::PaddedFilters() );
static auto dilated_filters = ::testing::ValuesIn( nn::test::filt_gen::DilatedFilters() );


INSTANTIATE_TEST_SUITE_P(Simple, WindowLocationTest, simple_filters);
INSTANTIATE_TEST_SUITE_P(Padded, WindowLocationTest, padded_filters);
INSTANTIATE_TEST_SUITE_P(Dilated, WindowLocationTest, dilated_filters);


