
#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>
#include <cassert>
#include <limits>

#include "gtest/gtest.h"

#include "OutputTransformFn.hpp"
#include "Rand.hpp"

#include "VpuHelpers.hpp"


class ShiftInt8OutputTransformParamsTest : public ::testing::TestWithParam<int> {};
class ShiftInt8OutputTransformTest : public ::testing::TestWithParam<int> {};


TEST_P(ShiftInt8OutputTransformParamsTest, ConstructorA)
{
  const int img_channels = GetParam();
  auto rand = nn::test::Rand(img_channels * 34563);

  for(int k = 0; k < 10; k++){
    int16_t shift = rand.rand<int16_t>();
    auto params = ShiftInt8OutputTransform::Params(img_channels, shift);
    ASSERT_EQ(img_channels, params.output_img_channels);
    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++)
      ASSERT_EQ(shift, params.shifts[i]);
  }

}

TEST_P(ShiftInt8OutputTransformParamsTest, ConstructorB)
{
  const int img_channels = GetParam();
  auto rand = nn::test::Rand(img_channels * 43);

  for(int k = 0; k < 10; k++){
    auto geom = nn::ImageGeometry(rand.rand(1, 100), rand.rand(1, 100), img_channels);
    int16_t shift = rand.rand<int16_t>();
    auto params = ShiftInt8OutputTransform::Params(geom, shift);
    ASSERT_EQ(img_channels, params.output_img_channels) << "nn::ImageGeometry: " << geom;
    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++)
      ASSERT_EQ(shift, params.shifts[i]);
  }
}


TEST_P(ShiftInt8OutputTransformParamsTest, Serialization)
{
  /// @TODO: astew: This doesn't currently make any considerations for endianness.

  int32_t img_channels = GetParam();
  auto rand = nn::test::Rand(img_channels * 7754);

  auto stream = std::stringstream();

  for(int k = 0; k < 10; k++){
    int16_t shift = rand.rand<int16_t>(0, 20);
    auto params = ShiftInt8OutputTransform::Params(img_channels, shift);
    params.Serialize(stream);

    params = ShiftInt8OutputTransform::Params(stream);
    ASSERT_EQ(img_channels, params.output_img_channels);
    
    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++)
      ASSERT_EQ(shift, params.shifts[i]);
  }
}





TEST_P(ShiftInt8OutputTransformTest, ConstructorA)
{
  auto rand = nn::test::Rand(GetParam() * 5656);
  int16_t shift = rand.rand<int16_t>();

  auto params = ShiftInt8OutputTransform::Params(GetParam(), shift);

  auto ot = ShiftInt8OutputTransform(&params);

  ASSERT_EQ(ot.params->output_img_channels, GetParam());
  ASSERT_EQ(ot.params, &params);
  for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++)
    ASSERT_EQ(shift, params.shifts[i]);
}


TEST_P(ShiftInt8OutputTransformTest, output_transform_fn)
{
  auto rand = nn::test::Rand(GetParam() * 6422);

  auto shift = rand.rand<int16_t>(0, 10);

  auto params = ShiftInt8OutputTransform::Params(GetParam(), shift);
  auto ot = ShiftInt8OutputTransform(&params);

  auto out = std::vector<int8_t>( ShiftInt8OutputTransform::ChannelsPerOutputGroup + 32);
  auto exp = std::vector<int8_t>( ShiftInt8OutputTransform::ChannelsPerOutputGroup );

  auto acc32 = std::vector<int32_t>( VPU_INT16_ACC_PERIOD );

  auto out_base = &out[16];

  const auto cog_count = (GetParam() + ShiftInt8OutputTransform::ChannelsPerOutputGroup - 1) 
                            / ShiftInt8OutputTransform::ChannelsPerOutputGroup;


  for(int k = 0; k < cog_count; k++){

    // Number of channels that should be written this iteration
    int first_chan = ShiftInt8OutputTransform::ChannelsPerOutputGroup * k;
    int should_write = params.output_img_channels - first_chan;
    should_write = std::min(should_write, ShiftInt8OutputTransform::ChannelsPerOutputGroup);

    // Clear the test image and our expectation
    std::memset(&out[0], 0, sizeof(int8_t) * out.size());

    vpu_ring_buffer_t acc;

    // Fill accumulator with random data
    for(int i = 0; i < VPU_INT16_ACC_PERIOD; i++){

      // Easiest if we work backwards. Choose the value we'd like to end up with. We'll actually go a little
      // beyond the int8 range so that we can test saturation, too
      const int res = rand.rand<int>(INT8_MIN - 5, INT8_MAX + 5);

      // Now left-shift to figure out what the accumulator should have been
      acc32[i] = res << shift;

      // To test the rounding behavior we'll also give it a nudge
      if(shift > 0){
        int32_t max_nudge = (1 << (shift-1))-1;
        int32_t min_nudge = -(1 << (shift-1));
        
        acc32[i] = acc32[i] + (rand.rand<bool>()? max_nudge : min_nudge);
      }

      // Set the expectation
      exp[i] = (i < should_write)? nn::test::vpu_sat<int8_t>(res, false) : 0;

      // Convert that to the split 32-bit accumulator
      nn::test::split_acc32(acc.vD[i], acc.vR[i], acc32[i]);
    }

    auto next_Y = ot.output_transform_fn(out_base, &acc, k);

#define EXTRA_OUT   "Extra Failure Details...\n"                                                        \
                    "  k = " << k << "\t\t(output channel group)\n"                                       \
                    "  first_chan = " << first_chan << "\n"                                             \
                    "  should_write = " << should_write << "\t\t(number of channels to be written)\n"     \
                    "  i = " << i << "\t\t(index relative to out_base[])\n"

    ASSERT_EQ(next_Y, &out_base[should_write]);

    // Everything before out_base should be zeros
    for(int i = 0; i < 16; i++)
      ASSERT_EQ(0, out[i]) << EXTRA_OUT;
    
    // Everything after out_base[should_write] should be zeros
    for(int i = 0; i < 16; i++)
      ASSERT_EQ(0, out[16 + ShiftInt8OutputTransform::ChannelsPerOutputGroup + i]) << EXTRA_OUT;
    
    // We should be expecting zeros for any output channel tail
    for(int i = should_write; i < ShiftInt8OutputTransform::ChannelsPerOutputGroup; i++)
      ASSERT_EQ(exp[i], out_base[i]) << EXTRA_OUT;
    
    // And these should be equal to the expectation.
    for(int i = 0; i < should_write; i++)
      ASSERT_EQ(exp[i], out_base[i]) << EXTRA_OUT
        << "  exp[" << i << "] = 0x" << std::hex << unsigned(uint8_t(exp[i])) 
            << std::dec << " (" << int(exp[i]) << ")\n"
        << "  out_base[" << i << "] = 0x" << std::hex << unsigned(uint8_t(out_base[i])) 
            << std::dec << " (" << int(out_base[i]) << ")\n"
        << "  acc32[" << i << "] = 0x" << std::hex << uint32_t(acc32[i]) << std::dec << " (" << acc32[i] << ")"
          << "\t(32-bit accumulator value prior to shift)\n"
        << "  acc.vD[" << i << "] = 0x" << std::hex << uint16_t(acc.vD[i]) << std::dec << " (" << acc.vD[i] << ")\n"
        << "  acc.vR[" << i << "] = 0x" << std::hex << uint16_t(acc.vR[i]) << std::dec << " (" << acc.vR[i] << ")\n"
        << "  shifts[first_chan+i] = shifts[" << (first_chan+i) << "] = " << shift << "\n";

#undef EXTRA_OUT
  }
}


INSTANTIATE_TEST_SUITE_P(, ShiftInt8OutputTransformParamsTest, ::testing::Range(4, 80, 4));
INSTANTIATE_TEST_SUITE_P(, ShiftInt8OutputTransformTest, ::testing::Range(4, 80, 4));







TEST(shift_int8_output_transform_ref,shift_int8_output_transform_ref)
{
  auto rand = nn::test::Rand(4635);

  auto shifts = std::vector<int16_t>(VPU_INT8_ACC_PERIOD);

  auto out = std::vector<int8_t>( VPU_INT8_ACC_PERIOD );
  auto exp = std::vector<int8_t>( VPU_INT8_ACC_PERIOD );

  auto acc32 = std::vector<int32_t>( VPU_INT8_ACC_PERIOD );

  for(int k = 0; k < 1000; k++){

    for(int write_chans = 0; write_chans <= VPU_INT8_ACC_PERIOD; write_chans++){
      std::memset(&out[0], 0, sizeof(int8_t) * out.size());
      vpu_ring_buffer_t acc;

      auto shift = rand.rand<int16_t>(0, 10);

      for(int i = 0; i < VPU_INT16_ACC_PERIOD; i++){
        const int res = rand.rand<int>(INT8_MIN - 5, INT8_MAX + 5);

        shifts[i] = shift;
        acc32[i] = res << shifts[i];
        if(shifts[i] > 0){
          int32_t max_nudge = (1 << (shifts[i]-1))-1;
          int32_t min_nudge = -(1 << (shifts[i]-1));
          acc32[i] = acc32[i] + (rand.rand<bool>()? max_nudge : min_nudge);
        }
        exp[i] = (i < write_chans)? nn::test::vpu_sat<int8_t>(res, false) : 0;
        nn::test::split_acc32(acc.vD[i], acc.vR[i], acc32[i]);
      }

      shift_int8_output_transform_ref(&out[0], &acc, &shifts[0], write_chans);
      
      for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++)
        ASSERT_EQ(exp[i], out[i]);
    }
  }
}
