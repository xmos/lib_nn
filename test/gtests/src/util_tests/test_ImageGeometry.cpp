
#include <cstring>
#include <iostream>
#include <tuple>
#include <vector>

#include "Rand.hpp"
#include "geom/ImageGeometry.hpp"
#include "gtest/gtest.h"

using namespace nn;

/**
 * Generates a set of ImageGeometry objects used as parameters for tests.
 * It's templated because the GetElement() method is templated, and I don't know
 * how to write the GetElement() test in a way that works for more than one
 * template type (and isn't ugly)
 */
template <typename T>
static std::vector<ImageGeometry> TestGeometries() {
  const auto max_height = 4;
  const auto max_width = 4;
  const auto max_depth = 8;

  auto res = std::vector<ImageGeometry>();

  for (int h = 1; h < max_height; ++h)
    for (int w = 1; w < max_width; ++w)
      for (int d = 1; d < max_depth; ++d)
        res.push_back(ImageGeometry(h, w, d, sizeof(T)));

  return res;
}

/**
 * Test Classes
 *
 * (I wouldn't need to specify more than one class here, except that when
 * running parameterized tests, ALL parameters are run against ALL tests of that
 * class. Unfortunate.)
 */

/** ImageGeometryTest is run against all three types */
class ImageGeometry_Test : public ::testing::TestWithParam<ImageGeometry> {};

/**
 * Tests
 */

/////////////////////////////////////////////////////////////////////////
//
//
TEST_P(ImageGeometry_Test, PixelCount) {
  auto img = GetParam();
  ASSERT_EQ(img.width * img.height, img.PixelCount());
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST_P(ImageGeometry_Test, ElementCounts) {
  auto img = GetParam();
  ASSERT_EQ(img.depth, img.PixelElements());
  ASSERT_EQ(img.depth * img.width, img.RowElements());
  ASSERT_EQ(img.depth * img.height, img.ColElements());
  ASSERT_EQ(img.depth * img.width * img.height, img.ElementCount());
  ASSERT_EQ(img.depth * img.width * img.height * img.depth,
            img.VolumeElements());
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST_P(ImageGeometry_Test, ByteCounts) {
  auto img = GetParam();
  ASSERT_EQ(img.depth * img.channel_depth, img.PixelBytes());
  ASSERT_EQ(img.depth * img.width * img.channel_depth, img.RowBytes());
  ASSERT_EQ(img.depth * img.height * img.channel_depth, img.ColBytes());
  ASSERT_EQ(img.depth * img.width * img.height * img.channel_depth,
            img.ImageBytes());
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST_P(ImageGeometry_Test, Index) {
  auto img = GetParam();

  int k = 0;
  for (int row = 0; row < img.height; ++row) {
    for (int col = 0; col < img.width; ++col) {
      for (int chan = 0; chan < img.depth; ++chan) {
        ASSERT_EQ(k, img.Index(row, col, chan));
        ASSERT_EQ(k, img.Index({row, col, chan}));
        k++;
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST_P(ImageGeometry_Test, GetStride) {
  auto img = GetParam();

  // ImageBytes() so it is deterministic but not the same for every case
  auto rng = nn::test::Rand(img.ImageBytes());

  for (int k = 0; k < 10; ++k) {
    auto rows1 = rng.rand<unsigned>(0, img.height - 1);
    auto cols1 = rng.rand<unsigned>(0, img.width - 1);
    auto xans1 = rng.rand<unsigned>(0, img.depth - 1);
    auto vect1 = ImageVect(rows1, cols1, xans1);

    auto rows2 = rng.rand<unsigned>(0, img.height - 1);
    auto cols2 = rng.rand<unsigned>(0, img.width - 1);
    auto xans2 = rng.rand<unsigned>(0, img.depth - 1);
    auto vect2 = ImageVect(rows2, cols2, xans2);

    auto stride = img.GetStride(rows1, cols1, xans1);

    ASSERT_EQ(stride,
              (xans1 + cols1 * img.depth + rows1 * img.depth * img.width) *
                  img.channel_depth);

    stride = img.GetStride(vect1);

    ASSERT_EQ(stride,
              (xans1 + cols1 * img.depth + rows1 * img.depth * img.width) *
                  img.channel_depth);

    stride = img.GetStride(vect1, vect2);

    auto delta = vect2 - vect1;

    ASSERT_EQ(stride, (delta.channel + delta.col * img.depth +
                       delta.row * img.depth * img.width) *
                          img.channel_depth);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST_P(ImageGeometry_Test, IsWithinImage) {
  auto img = GetParam();

  for (int row = 0; row < img.height; ++row) {
    for (int col = 0; col < img.width; ++col) {
      for (int chan = 0; chan < img.depth; ++chan) {
        EXPECT_TRUE(img.IsWithinImage(ImageVect(row, col, chan)));
        EXPECT_TRUE(img.IsWithinImage(row, col, chan));
      }
    }
  }

  for (int row = -2; row <= 2; ++row) {
    for (int col = -2; col <= 2; ++col) {
      for (int chan = -2; chan <= 2; ++chan) {
        if (row == 0 && col == 0 && chan == 0) continue;

        int xr = row + ((row <= 0) ? 0 : (int(img.height) - 1));
        int xc = col + ((col <= 0) ? 0 : (int(img.width) - 1));
        int xx = chan + ((chan <= 0) ? 0 : (int(img.depth) - 1));

        EXPECT_FALSE(img.IsWithinImage(ImageVect(xr, xc, xx)));
        EXPECT_FALSE(img.IsWithinImage(xr, xc, xx));
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
template <typename T_elm>
static void _ElementTest(nn::ImageGeometry img) {
  img.channel_depth = sizeof(T_elm);
  auto input = std::vector<T_elm>(img.ElementCount());
  int k = 0;
  for (int row = 0; row < img.height; ++row) {
    for (int col = 0; col < img.width; ++col) {
      for (int chan = 0; chan < img.depth; ++chan) {
        T_elm& refA = img.Element<T_elm>(&input[0], row, col, chan);
        T_elm& refB = input[k++];
        ASSERT_EQ(&(refA), &(refB));
        refA = T_elm(13 * row + 7 * col + chan);
        ASSERT_EQ(refB, T_elm(13 * row + 7 * col + chan));
      }
    }
  }
}

TEST_P(ImageGeometry_Test, Element) {
  switch (GetParam().channel_depth) {
    case 1:
      ASSERT_NO_FATAL_FAILURE(_ElementTest<int8_t>(GetParam()));
      break;
    case 2:
      ASSERT_NO_FATAL_FAILURE(_ElementTest<int16_t>(GetParam()));
      break;
    case 4:
      ASSERT_NO_FATAL_FAILURE(_ElementTest<int32_t>(GetParam()));
      break;
    default:
      FAIL();
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
template <typename T_elm>
static void _GetTest(nn::ImageGeometry img) {
  img.channel_depth = sizeof(T_elm);
  auto input = std::vector<T_elm>(img.ElementCount());
  int k = 0;
  for (int row = -1; row <= img.height; ++row) {
    for (int col = -1; col <= img.width; ++col) {
      for (int xan = -1; xan <= img.depth; ++xan) {
        ImageVect v(row, col, xan);
#define FAIL_MSG "v = " << v << " | k = " << k
        if (img.IsWithinImage(v)) {
          T_elm& elm = input[k++];
          elm = -23;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -23) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -23)
              << FAIL_MSG;
          elm = 77;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, 99), 77) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, 99), 77)
              << FAIL_MSG;
        } else {
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -52) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -52)
              << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, 99), 99) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, 99), 99)
              << FAIL_MSG;
        }
#undef FAIL_MSG
      }
    }
  }
}

TEST_P(ImageGeometry_Test, Get) {
  switch (GetParam().channel_depth) {
    case 1:
      ASSERT_NO_FATAL_FAILURE(_GetTest<int8_t>(GetParam()));
      break;
    case 2:
      ASSERT_NO_FATAL_FAILURE(_GetTest<int16_t>(GetParam()));
      break;
    case 4:
      ASSERT_NO_FATAL_FAILURE(_GetTest<int32_t>(GetParam()));
      break;
    default:
      FAIL();
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
template <typename T_elm>
static void _ApplyOpTest(nn::ImageGeometry img) {
  img.channel_depth = sizeof(T_elm);
  auto buff = std::vector<T_elm>(img.ElementCount());
  std::memset(&buff[0], 0, buff.size() * sizeof(T_elm));

  auto lam = [](const int row, const int col, const int chan, T_elm& elm) {
    elm = 1;
  };

  img.ApplyOperation<T_elm>(&buff[0], lam);

  for (int k = 0; k < buff.size(); k++) {
    ASSERT_EQ(int(1), int(buff[k]));
  }
}

TEST_P(ImageGeometry_Test, ApplyOperation) {
  switch (GetParam().channel_depth) {
    case 1:
      ASSERT_NO_FATAL_FAILURE(_ApplyOpTest<int8_t>(GetParam()));
      break;
    case 2:
      ASSERT_NO_FATAL_FAILURE(_ApplyOpTest<int16_t>(GetParam()));
      break;
    case 4:
      ASSERT_NO_FATAL_FAILURE(_ApplyOpTest<int32_t>(GetParam()));
      break;
    default:
      FAIL();
  }
}

INSTANTIATE_TEST_SUITE_P(int8, ImageGeometry_Test,
                         ::testing::ValuesIn(TestGeometries<int8_t>()));
INSTANTIATE_TEST_SUITE_P(int16, ImageGeometry_Test,
                         ::testing::ValuesIn(TestGeometries<int16_t>()));
INSTANTIATE_TEST_SUITE_P(int32, ImageGeometry_Test,
                         ::testing::ValuesIn(TestGeometries<int32_t>()));
