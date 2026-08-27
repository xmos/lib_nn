// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <vector>

#include "FilterGeometryIterHelper.hpp"
#include "Rand.hpp"
#include "geom/WindowLocation.hpp"
#include "unity.h"
#include "unity_fixture.h"

using namespace nn;

TEST_GROUP(group_WindowLocation);
TEST_SETUP(group_WindowLocation) {}
TEST_TEAR_DOWN(group_WindowLocation) {}
TEST_GROUP_RUNNER(group_WindowLocation) {
  RUN_TEST_CASE(group_WindowLocation, InputStart);
  RUN_TEST_CASE(group_WindowLocation, InputEnd);
  RUN_TEST_CASE(group_WindowLocation, InputCoords);
  RUN_TEST_CASE(group_WindowLocation, Padding);
  RUN_TEST_CASE(group_WindowLocation, SignedPadding);
  RUN_TEST_CASE(group_WindowLocation, IsPadding);
  RUN_TEST_CASE(group_WindowLocation, InputElement);
  RUN_TEST_CASE(group_WindowLocation, GetInput);
  RUN_TEST_CASE(group_WindowLocation, InputIndex);
  RUN_TEST_CASE(group_WindowLocation, Fold);
}

static nn::ff::FilterGeometryIterator filter_sets[] = {
    test::unpadded::SimpleDepthwise({1, 8}, {1, 4}, {4, 66}),
};

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_WindowLocation, InputStart) {
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

            TEST_ASSERT_EQUAL(start.row, exp.row);
            TEST_ASSERT_EQUAL(start.col, exp.col);
            TEST_ASSERT_EQUAL(start.channel, exp.channel);

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
TEST(group_WindowLocation, InputEnd) {
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

            TEST_ASSERT_EQUAL(end.row, exp.row);
            TEST_ASSERT_EQUAL(end.col, exp.col);
            TEST_ASSERT_EQUAL(end.channel, exp.channel);

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
TEST(group_WindowLocation, InputCoords) {
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

                  TEST_ASSERT_EQUAL(in_coords.row, exp.row);
                  TEST_ASSERT_EQUAL(in_coords.col, exp.col);
                  TEST_ASSERT_EQUAL(in_coords.channel, exp.channel);

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
TEST(group_WindowLocation, Padding) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          auto loc = WindowLocation(filter, ImageVect(yr, yc, 0));
          auto actual = loc.Padding();

          for (int kr = 0; kr < filter.window.shape.height; kr++) {
            for (int kc = 0; kc < filter.window.shape.width; kc++) {
              TEST_ASSERT_EQUAL(loc.IsPadding(kr, kc, 0),
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
TEST(group_WindowLocation, SignedPadding) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      for (int yr = 0; yr < filter.output.height; yr++) {
        for (int yc = 0; yc < filter.output.width; yc++) {
          auto loc = WindowLocation(filter, ImageVect(yr, yc, 0));

          auto pad = loc.SignedPadding();

          auto p = loc.InputStart();
          TEST_ASSERT_TRUE(filter.input.IsWithinImage(p.add(pad.top - 0, 0, 0)));
          TEST_ASSERT_FALSE(filter.input.IsWithinImage(p.add(pad.top - 1, 0, 0)));

          TEST_ASSERT_TRUE(filter.input.IsWithinImage(p.add(0, pad.left - 0, 0)));
          TEST_ASSERT_FALSE(filter.input.IsWithinImage(p.add(0, pad.left - 1, 0)));

          p = loc.InputEnd();
          TEST_ASSERT_TRUE(filter.input.IsWithinImage(p.add(-pad.bottom + 0, 0, 0)));
          TEST_ASSERT_FALSE(
              filter.input.IsWithinImage(p.add(-pad.bottom + 1, 0, 0)));

          TEST_ASSERT_TRUE(filter.input.IsWithinImage(p.add(0, -pad.right + 0, 0)));
          TEST_ASSERT_FALSE(filter.input.IsWithinImage(p.add(0, -pad.right + 1, 0)));
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_WindowLocation, IsPadding) {
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

                  TEST_ASSERT_EQUAL(expected, actual);
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
TEST(group_WindowLocation, InputElement) {
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

                  TEST_ASSERT_EQUAL_PTR(expected, actual);
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
TEST(group_WindowLocation, GetInput) {
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

                  TEST_ASSERT_EQUAL(expected, actual);
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
TEST(group_WindowLocation, InputIndex) {
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
                  TEST_ASSERT_EQUAL(offset, index);
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
TEST(group_WindowLocation, Fold) {
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

            int32_t expected = 123;

            int32_t original_expected = expected;

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

            auto res = loc.Fold<int32_t, int8_t>(&input_img[0], lfunc,
                                                 original_expected, 0);

            TEST_ASSERT_EQUAL(expected, res);
          }
        }
      }
    }
  }
}
