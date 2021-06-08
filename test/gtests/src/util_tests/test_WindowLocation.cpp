
#include <iostream>
#include <tuple>
#include <vector>

#include "FilterGen.hpp"
#include "FilterGeometryIterHelper.hpp"
#include "Rand.hpp"
#include "geom/WindowLocation.hpp"
#include "gtest/gtest.h"

using namespace nn;

// class WindowLocationTest : public ::testing::TestWithParam<Filter2dGeometry>
// {};

static nn::ff::FilterGeometryIterator filter_sets[] = {
    test::unpadded::SimpleDepthwise({1, 8}, {1, 4}, {4, 66}),
};

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, InputStart) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      ImageVect exp(0, 0, 0);

      exp.row = filter.window.start.row;
      for (int yr = 0; yr < filter.output.height; yr++) {
        exp.col = filter.window.start.col;
        for (int yc = 0; yc < filter.output.width; yc++) {
          exp.channel = 0;
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            auto start = loc.InputStart();

            EXPECT_EQ(start.row, exp.row)
                << "Output Coords: " << loc.output_coords;
            EXPECT_EQ(start.col, exp.col)
                << "Output Coords: " << loc.output_coords;
            EXPECT_EQ(start.channel, exp.channel)
                << "Output Coords: " << loc.output_coords;

            exp.channel += filter.window.stride.channel;
          }

          exp.col += filter.window.stride.col;
        }

        exp.row += filter.window.stride.row;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, InputEnd) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      ImageVect exp(0, 0, 0);

      exp.row = filter.window.start.row +
                (filter.window.shape.height - 1) * filter.window.dilation.row;
      for (int yr = 0; yr < filter.output.height; yr++) {
        exp.col = filter.window.start.col +
                  (filter.window.shape.width - 1) * filter.window.dilation.col;
        for (int yc = 0; yc < filter.output.width; yc++) {
          exp.channel = filter.window.shape.depth - 1;
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            auto end = loc.InputEnd();

            EXPECT_EQ(end.row, exp.row)
                << "Output Coords: " << loc.output_coords;
            EXPECT_EQ(end.col, exp.col)
                << "Output Coords: " << loc.output_coords;
            EXPECT_EQ(end.channel, exp.channel)
                << "Output Coords: " << loc.output_coords;

            exp.channel += filter.window.stride.channel;
          }

          exp.col += filter.window.stride.col;
        }

        exp.row += filter.window.stride.row;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, InputCoords) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      ImageVect exp(0, 0, 0);

      exp.row = filter.window.start.row +
                (filter.window.shape.height - 1) * filter.window.dilation.row;
      for (int yr = 0; yr < filter.output.height; yr++) {
        exp.col = filter.window.start.col +
                  (filter.window.shape.width - 1) * filter.window.dilation.col;
        for (int yc = 0; yc < filter.output.width; yc++) {
          exp.channel = filter.window.shape.depth - 1;
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            auto start = loc.InputStart();
            auto exp = start;

            for (int kr = 0; kr < filter.window.shape.height; kr++) {
              exp.col = start.col;
              for (int kc = 0; kc < filter.window.shape.width; kc++) {
                exp.channel = start.channel;
                for (int kx = 0; kx < filter.window.shape.depth; kx++) {
                  auto in_coords = loc.InputCoords(kr, kc, kx);

                  EXPECT_EQ(in_coords.row, exp.row)
                      << "Output Coords: " << loc.output_coords
                      << "| Filter Coords: " << ImageVect(kr, kc, kx);
                  EXPECT_EQ(in_coords.col, exp.col)
                      << "Output Coords: " << loc.output_coords
                      << "| Filter Coords: " << ImageVect(kr, kc, kx);
                  EXPECT_EQ(in_coords.channel, exp.channel)
                      << "Output Coords: " << loc.output_coords
                      << "| Filter Coords: " << ImageVect(kr, kc, kx);

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
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, Padding) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          auto loc = WindowLocation(filter, ImageVect(yr, yc, 0));
          auto actual = loc.Padding();

          for (int kr = 0; kr < filter.window.shape.height; kr++) {
            for (int kc = 0; kc < filter.window.shape.width; kc++) {
              EXPECT_EQ(loc.IsPadding(kr, kc, 0),
                        (kr < actual.top) ||
                            (kr > filter.window.shape.height - actual.bottom) ||
                            (kc < actual.left) ||
                            (kc > filter.window.shape.width - actual.right));
            }
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, SignedPadding) {
  auto rand = nn::test::Rand(981513);

  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          auto loc = WindowLocation(filter, ImageVect(yr, yc, 0));

          auto pad = loc.SignedPadding();

          auto p = loc.InputStart();
          ASSERT_TRUE(filter.input.IsWithinImage(p.add(pad.top - 0, 0, 0)));
          ASSERT_FALSE(filter.input.IsWithinImage(p.add(pad.top - 1, 0, 0)));

          ASSERT_TRUE(filter.input.IsWithinImage(p.add(0, pad.left - 0, 0)));
          ASSERT_FALSE(filter.input.IsWithinImage(p.add(0, pad.left - 1, 0)));

          p = loc.InputEnd();
          ASSERT_TRUE(filter.input.IsWithinImage(p.add(-pad.bottom + 0, 0, 0)));
          ASSERT_FALSE(
              filter.input.IsWithinImage(p.add(-pad.bottom + 1, 0, 0)));

          ASSERT_TRUE(filter.input.IsWithinImage(p.add(0, -pad.right + 0, 0)));
          ASSERT_FALSE(filter.input.IsWithinImage(p.add(0, -pad.right + 1, 0)));
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, IsPadding) {
  auto rand = nn::test::Rand(4564523);

  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      auto input_img = std::vector<int8_t>(filter.input.ElementCount());

      for (int k = 0; k < input_img.size(); k++) {
        input_img[k] = rand.rand<int8_t>();
      }

      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            for (int kr = 0; kr < filter.window.shape.height; kr++) {
              for (int kc = 0; kc < filter.window.shape.width; kc++) {
                for (int kx = 0; kx < filter.window.shape.depth; kx++) {
                  auto in_coords = loc.InputCoords(kr, kc, kx);

                  auto expected = !filter.input.IsWithinImage(in_coords);
                  auto actual = loc.IsPadding(kr, kc, kx);

                  EXPECT_EQ(expected, actual)
                      << "Output Coords: " << loc.output_coords
                      << "| Filter Coords: " << ImageVect(kr, kc, kx);
                }
              }
            }
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, InputElement) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      auto input_img = std::vector<int8_t>(filter.input.ElementCount());

      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            for (int kr = 0; kr < filter.window.shape.height; kr++) {
              if (loc.IsPadding(kr, 0)) continue;

              for (int kc = 0; kc < filter.window.shape.width; kc++) {
                if (loc.IsPadding(kr, kc)) continue;

                for (int kx = 0; kx < filter.window.shape.depth; kx++) {
                  auto in_coords = loc.InputCoords(kr, kc, kx);

                  int index = (in_coords.row *
                               int(filter.input.depth * filter.input.width)) +
                              (in_coords.col * int(filter.input.depth)) +
                              (in_coords.channel);

                  int8_t* expected = &input_img[index];
                  int8_t* actual = &loc.InputElement(&input_img[0], kr, kc, kx);

                  EXPECT_EQ(expected, actual)
                      << "Output Coords: " << loc.output_coords
                      << " | Filter Coords: " << ImageVect(kr, kc, kx)
                      << " | in_coords: " << in_coords << " | index: " << index
                      << " | input_img: " << &input_img[0];
                }
              }
            }
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, GetInput) {
  auto rand = nn::test::Rand(754444);

  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      auto input_img = std::vector<int8_t>(filter.input.ElementCount());

      for (int k = 0; k < input_img.size(); k++) {
        input_img[k] = rand.rand<int8_t>();
      }

      auto zero_pad = rand.rand<int8_t>();

      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            for (int kr = 0; kr < filter.window.shape.height; kr++) {
              for (int kc = 0; kc < filter.window.shape.width; kc++) {
                for (int kx = 0; kx < filter.window.shape.depth; kx++) {
                  auto in_coords = loc.InputCoords(kr, kc, kx);

                  int index = (in_coords.row *
                               int(filter.input.depth * filter.input.width)) +
                              (in_coords.col * int(filter.input.depth)) +
                              (in_coords.channel);

                  int8_t expected =
                      loc.IsPadding(kr, kc, kx) ? zero_pad : input_img[index];
                  int8_t actual =
                      loc.GetInput<int8_t>(&input_img[0], kr, kc, kx, zero_pad);

                  ASSERT_EQ(expected, actual)
                      << "Output Coords: " << loc.output_coords
                      << " | Filter Coords: " << ImageVect(kr, kc, kx)
                      << " | in_coords: " << in_coords << " | index: " << index
                      << " | input_img: " << &input_img[0]
                      << " | zero_pad: " << int(zero_pad);
                }
              }
            }
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, InputIndex) {
  auto rand = nn::test::Rand(7695699);

  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      auto input_img = std::vector<int8_t>(filter.input.ElementCount());

      for (int k = 0; k < input_img.size(); k++) {
        input_img[k] = rand.rand<int8_t>();
      }

      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            for (int kr = 0; kr < filter.window.shape.height; kr++) {
              for (int kc = 0; kc < filter.window.shape.width; kc++) {
                for (int kx = 0; kx < filter.window.shape.depth; kx++) {
                  auto in_coords = loc.InputCoords(kr, kc, kx);
                  auto offset = filter.input.GetStride(in_coords);
                  auto index = loc.InputIndex(kr, kc, kx);
                  EXPECT_EQ(offset, index);
                }
              }
            }
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(WindowLocation_Test, Fold) {
  auto rand = nn::test::Rand(4564523);

  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      auto input_img = std::vector<int8_t>(filter.input.ElementCount());

      for (int k = 0; k < input_img.size(); k++) {
        input_img[k] = rand.rand<int8_t>();
      }

      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          for (int yx = 0; yx < filter.output.depth; yx++) {
            auto loc = WindowLocation(filter, ImageVect(yr, yc, yx));

            int32_t expected = 1234;

            for (int kr = 0; kr < filter.window.shape.height; kr++) {
              for (int kc = 0; kc < filter.window.shape.width; kc++) {
                for (int kx = 0; kx < filter.window.shape.depth; kx++) {
                  auto input =
                      loc.GetInput<int8_t>(&input_img[0], kr, kc, kx, 0);
                  if (input == 0)
                    expected++;
                  else
                    expected *= input;
                }
              }
            }

            auto lfunc = [](const ImageVect&, const ImageVect&,
                            const int32_t acc, const int8_t elm,
                            const bool) -> int32_t {
              if (elm == 0) return (acc + 1);
              return acc * elm;
            };

            auto res = loc.Fold<int32_t, int8_t>(&input_img[0], lfunc, 1234, 0);

            ASSERT_EQ(expected, res);
          }
        }
      }
    }
  }
}
