// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <cstring>
#include <vector>

#include "Rand.hpp"
#include "geom/ImageGeometry.hpp"
#include "unity.h"
#include "unity_fixture.h"

using namespace nn;

TEST_GROUP(group_ImageGeometry);
TEST_SETUP(group_ImageGeometry) {}
TEST_TEAR_DOWN(group_ImageGeometry) {}
TEST_GROUP_RUNNER(group_ImageGeometry) {
  RUN_TEST_CASE(group_ImageGeometry, PixelCount);
  RUN_TEST_CASE(group_ImageGeometry, ElementCounts);
  RUN_TEST_CASE(group_ImageGeometry, ByteCounts);
  RUN_TEST_CASE(group_ImageGeometry, Index);
  RUN_TEST_CASE(group_ImageGeometry, GetStride);
  RUN_TEST_CASE(group_ImageGeometry, IsWithinImage);
  RUN_TEST_CASE(group_ImageGeometry, Element);
  RUN_TEST_CASE(group_ImageGeometry, Get);
  RUN_TEST_CASE(group_ImageGeometry, ApplyOperation);
}

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
        res.push_back(ImageGeometry(h, w, d, sizeof(T) * CHAR_BIT));

  return res;
}

/**
 * All geometries across every element type this suite covers, matching
 * gtest's original behaviour of running every test case against every
 * int8/int16/int32 parameter set.
 */
static std::vector<ImageGeometry> AllTestGeometries() {
  auto res = TestGeometries<int8_t>();
  auto v16 = TestGeometries<int16_t>();
  auto v32 = TestGeometries<int32_t>();
  res.insert(res.end(), v16.begin(), v16.end());
  res.insert(res.end(), v32.begin(), v32.end());
  return res;
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageGeometry, PixelCount) {
  for (auto img : AllTestGeometries()) {
    TEST_ASSERT_EQUAL(img.width * img.height, img.PixelCount());
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageGeometry, ElementCounts) {
  for (auto img : AllTestGeometries()) {
    TEST_ASSERT_EQUAL(img.depth, img.PixelElements());
    TEST_ASSERT_EQUAL(img.depth * img.width, img.RowElements());
    TEST_ASSERT_EQUAL(img.depth * img.height, img.ColElements());
    TEST_ASSERT_EQUAL(img.depth * img.width * img.height, img.ElementCount());
    TEST_ASSERT_EQUAL(img.depth * img.width * img.height * img.depth,
              img.VolumeElements());
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageGeometry, ByteCounts) {
  for (auto img : AllTestGeometries()) {
    TEST_ASSERT_EQUAL(img.depth * img.element_bits / CHAR_BIT, img.PixelBytes());
    TEST_ASSERT_EQUAL(img.depth * img.width * img.element_bits / CHAR_BIT,
              img.RowBytes());
    TEST_ASSERT_EQUAL(img.depth * img.height * img.element_bits / CHAR_BIT,
              img.ColBytes());
    TEST_ASSERT_EQUAL(img.depth * img.width * img.height * img.element_bits / CHAR_BIT,
              img.ImageBytes());
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageGeometry, Index) {
  for (auto img : AllTestGeometries()) {
    int k = 0;
    for (int row = 0; row < img.height; ++row) {
      for (int col = 0; col < img.width; ++col) {
        for (int chan = 0; chan < img.depth; ++chan) {
          TEST_ASSERT_EQUAL(k, img.Index(row, col, chan));
          TEST_ASSERT_EQUAL(k, img.Index({row, col, chan}));
          k++;
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageGeometry, GetStride) {
  for (auto img : AllTestGeometries()) {
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

      TEST_ASSERT_EQUAL(stride * CHAR_BIT,
                (xans1 + cols1 * img.depth + rows1 * img.depth * img.width) *
                    img.element_bits);

      stride = img.GetStride(vect1);

      TEST_ASSERT_EQUAL(stride * CHAR_BIT,
                (xans1 + cols1 * img.depth + rows1 * img.depth * img.width) *
                    img.element_bits);

      stride = img.GetStride(vect1, vect2);

      auto delta = vect2 - vect1;

      TEST_ASSERT_EQUAL(stride * CHAR_BIT, (delta.channel + delta.col * img.depth +
                                    delta.row * img.depth * img.width) *
                                       img.element_bits);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageGeometry, IsWithinImage) {
  for (auto img : AllTestGeometries()) {
    for (int row = 0; row < img.height; ++row) {
      for (int col = 0; col < img.width; ++col) {
        for (int chan = 0; chan < img.depth; ++chan) {
          TEST_ASSERT_TRUE(img.IsWithinImage(ImageVect(row, col, chan)));
          TEST_ASSERT_TRUE(img.IsWithinImage(row, col, chan));
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

          TEST_ASSERT_FALSE(img.IsWithinImage(ImageVect(xr, xc, xx)));
          TEST_ASSERT_FALSE(img.IsWithinImage(xr, xc, xx));
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
template <typename T_elm>
static void _ElementTest(nn::ImageGeometry img) {
  img.element_bits = sizeof(T_elm) * CHAR_BIT;
  auto input = std::vector<T_elm>(img.ElementCount());
  int k = 0;
  for (int row = 0; row < img.height; ++row) {
    for (int col = 0; col < img.width; ++col) {
      for (int chan = 0; chan < img.depth; ++chan) {
        T_elm& refA = img.Element<T_elm>(&input[0], row, col, chan);
        T_elm& refB = input[k++];
        TEST_ASSERT_EQUAL_PTR(&(refB), &(refA));
        refA = T_elm(13 * row + 7 * col + chan);
        TEST_ASSERT_EQUAL(refB, T_elm(13 * row + 7 * col + chan));
      }
    }
  }
}

TEST(group_ImageGeometry, Element) {
  for (auto img : AllTestGeometries()) {
    switch (img.element_bits / CHAR_BIT) {
      case 1:
        _ElementTest<int8_t>(img);
        break;
      case 2:
        _ElementTest<int16_t>(img);
        break;
      case 4:
        _ElementTest<int32_t>(img);
        break;
      default:
        TEST_FAIL();
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
template <typename T_elm>
static void _GetTest(nn::ImageGeometry img) {
  img.element_bits = sizeof(T_elm) * CHAR_BIT;
  auto input = std::vector<T_elm>(img.ElementCount());
  int k = 0;
  for (int row = -1; row <= img.height; ++row) {
    for (int col = -1; col <= img.width; ++col) {
      for (int xan = -1; xan <= img.depth; ++xan) {
        ImageVect v(row, col, xan);
        if (img.IsWithinImage(v)) {
          T_elm& elm = input[k++];
          elm = -23;
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], v, -52), -23);
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], row, col, xan, -52), -23);
          elm = 77;
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], v, 99), 77);
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], row, col, xan, 99), 77);
        } else {
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], v, -52), -52);
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], row, col, xan, -52), -52);
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], v, 99), 99);
          TEST_ASSERT_EQUAL(img.Get<T_elm>(&input[0], row, col, xan, 99), 99);
        }
      }
    }
  }
}

TEST(group_ImageGeometry, Get) {
  for (auto img : AllTestGeometries()) {
    switch (img.element_bits / CHAR_BIT) {
      case 1:
        _GetTest<int8_t>(img);
        break;
      case 2:
        _GetTest<int16_t>(img);
        break;
      case 4:
        _GetTest<int32_t>(img);
        break;
      default:
        TEST_FAIL();
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
template <typename T_elm>
static void _ApplyOpTest(nn::ImageGeometry img) {
  img.element_bits = sizeof(T_elm) * CHAR_BIT;
  auto buff = std::vector<T_elm>(img.ElementCount());
  std::memset(&buff[0], 0, buff.size() * sizeof(T_elm));

  auto lam = [](const int row, const int col, const int chan, T_elm& elm) {
    elm = 1;
  };

  img.ApplyOperation<T_elm>(&buff[0], lam);

  for (int k = 0; k < buff.size(); k++) {
    TEST_ASSERT_EQUAL(int(1), int(buff[k]));
  }
}

TEST(group_ImageGeometry, ApplyOperation) {
  for (auto img : AllTestGeometries()) {
    switch (img.element_bits / CHAR_BIT) {
      case 1:
        _ApplyOpTest<int8_t>(img);
        break;
      case 2:
        _ApplyOpTest<int16_t>(img);
        break;
      case 4:
        _ApplyOpTest<int32_t>(img);
        break;
      default:
        TEST_FAIL();
    }
  }
}
