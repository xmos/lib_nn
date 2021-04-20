
#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>

#include "gtest/gtest.h"

#include "OutputTransformFn.hpp"
#include "Rand.hpp"

using namespace nn;

class DirectWriteOutputTransformParamsTest : public ::testing::TestWithParam<int> {};
class DirectWriteOutputTransformTest : public ::testing::TestWithParam<int> {};


TEST_P(DirectWriteOutputTransformParamsTest, ConstructorA)
{
  const int img_channels = GetParam();

  auto p = DirectWriteOutputTransform::Params(img_channels);

  ASSERT_EQ(img_channels, p.output_img_channels);

}

TEST_P(DirectWriteOutputTransformParamsTest, ConstructorB)
{
  const int img_channels = GetParam();

  auto rand = nn::test::Rand(img_channels * 79);

  for(int i = 0; i < 10; i++){
    auto geom = nn::ImageGeometry(rand.rand(1, 100), rand.rand(1, 100), img_channels);
    auto p = DirectWriteOutputTransform::Params(geom);
    ASSERT_EQ(img_channels, p.output_img_channels) << "nn::ImageGeometry: " << geom;
  }
}

TEST_P(DirectWriteOutputTransformParamsTest, ConstructorC)
{
  int32_t img_channels = GetParam();

  auto stream = std::stringstream();

  stream.write(reinterpret_cast<char*>(&img_channels), sizeof(img_channels));

  auto p = DirectWriteOutputTransform::Params(stream);

  ASSERT_EQ(img_channels, p.output_img_channels);

}


TEST_P(DirectWriteOutputTransformParamsTest, Serialize)
{
  /// @TODO: astew: This doesn't currently make any considerations for endianness.

  int32_t img_channels = GetParam();

  auto params = DirectWriteOutputTransform::Params(img_channels);

  auto stream = std::stringstream();

  params.Serialize(stream);

  int32_t v;

  stream.read(reinterpret_cast<char*>(&v), sizeof(v));

  ASSERT_EQ(img_channels, v);

}





TEST_P(DirectWriteOutputTransformTest, ConstructorA)
{
  auto params = DirectWriteOutputTransform::Params(GetParam());

  auto ot = DirectWriteOutputTransform(&params);

  ASSERT_EQ(ot.params->output_img_channels, GetParam());
  ASSERT_EQ(ot.params, &params);
}


TEST_P(DirectWriteOutputTransformTest, output_transform_fn)
{
  auto rand = nn::test::Rand(GetParam() * 1235);

  auto params = DirectWriteOutputTransform::Params(GetParam());
  auto ot = DirectWriteOutputTransform(&params);

  auto img = std::vector<int8_t>( DirectWriteOutputTransform::ChannelsPerOutputGroup + 32);
  auto exp = std::vector<int8_t>( DirectWriteOutputTransform::ChannelsPerOutputGroup + 32);

  auto pixel_base = &img[16];

  const auto cog_count = (GetParam() + DirectWriteOutputTransform::ChannelsPerOutputGroup - 1) 
                            / DirectWriteOutputTransform::ChannelsPerOutputGroup;

  for(int k = 0; k < cog_count; k++){
    // Number of channels that should be written this iteration
    int should_write = params.output_img_channels - DirectWriteOutputTransform::ChannelsPerOutputGroup * k;
    should_write = std::min(should_write, DirectWriteOutputTransform::ChannelsPerOutputGroup);

    // Clear the test image and our expectation
    std::memset(&img[0], 0, sizeof(int8_t) * img.size());
    std::memset(&exp[0], 0, sizeof(int8_t) * exp.size());

    // Fill accumulator with random data
    vpu_ring_buffer_t acc;
    for(int i = 0; i < VPU_INT16_ACC_PERIOD; i++){
      acc.vD[i] = rand.rand<int16_t>();
      acc.vR[i] = rand.rand<int16_t>();
    }

    // Determine out expectation
    for(int i = 0; i < should_write; i++){
      exp[16 + i] = reinterpret_cast<int8_t*>(acc.vR)[i];
    }

    auto next_Y = ot.output_transform_fn(pixel_base, &acc, k);

    ASSERT_EQ(next_Y, &pixel_base[should_write]);
    ASSERT_EQ(img, exp);

  }


}


INSTANTIATE_TEST_SUITE_P(, DirectWriteOutputTransformParamsTest, ::testing::Range(4, 80, 4));
INSTANTIATE_TEST_SUITE_P(, DirectWriteOutputTransformTest, ::testing::Range(4, 80, 4));


