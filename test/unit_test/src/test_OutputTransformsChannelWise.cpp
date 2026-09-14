// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <algorithm>
#include <cstdint>

#include "OutputTransformFn.hpp"

extern "C"
{

/*
group_output_transforms_channel_wise

for each channel c in parallel:
    accumulator = channelwise_saturate(accumulator, initial_shift[c])
    value = accumulator * multiplier[c] + bias[c]
    value = rounded_right_shift(value, final_shift)
    output[c] = saturate_to_int8(value)

*/

#include "unity.h"
#include "unity_fixture.h"

#include "etc/helpers.h"
#include "vpu_sim.h"

static void set_target(void)
{
#if defined(__VX4B__)
    SetNNTargetArch(TARGET_ARCH_VX4A);
#else
    SetNNTargetArch(TARGET_ARCH_XS3A);
#endif
}

static int8_t sat_int8(int val)
{
    val = val < INT8_MIN ? INT8_MIN : val;
    val = val > INT8_MAX ? INT8_MAX : val;
    return (int8_t)vpu_saturate_fixed(val, 8);
}

static int16_t get_multiplier()
{
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A
                                                  : VLMUL_SHR_VX4A;
    return sh == 15 ? INT16_MAX : (1 << sh);
}

TEST_GROUP(group_output_transforms_channel_wise);
TEST_SETUP(group_output_transforms_channel_wise)
{
    set_target();
}
TEST_TEAR_DOWN(group_output_transforms_channel_wise) {}
TEST_GROUP_RUNNER(group_output_transforms_channel_wise)
{
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_simple);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_simple)
{
    /*
    simple case:
        accumulator = [256, 512]
        initial_shift = [0, 0]
        multiplier = [1, 1]
        bias = [0, 0]
        final_shr = 0
    expected:
        channel 0: 256 / 256 = 1
        channel 1: 512 / 256 = 2
    */
    const int output_count = 2;
    const int16_t final_shr = 0;
    const int16_t multiplier = get_multiplier();
    const int8_t expected[output_count] = {1, 2};
    const int8_t expected_unchanged = 99;

    VPURingBuffer acc{};
    acc.SetAccu(0, 256);
    acc.SetAccu(1, 512);

    // The channelwise layout is [initial shifts][multipliers][biases].
    int16_t params_mem[output_count * 3] = {0, 0, multiplier, multiplier, 0, 0};
    int8_t out[output_count + 1] = {0, 0, expected_unchanged};

    nn::OT_int8_channelwise ot(output_count, final_shr);
    nn::otfn_int8_channelwise_params_t params = ot.getParams();
    int8_t *end = nn::otfn_int8_channelwise(&params, out, &acc, 0, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    for (int ch = 0; ch < output_count; ++ch)
    {
        TEST_ASSERT_EQUAL_INT8(expected[ch], out[ch]);
    }
    TEST_ASSERT_EQUAL_INT8(expected_unchanged, out[output_count]);
}

} // extern "C"
