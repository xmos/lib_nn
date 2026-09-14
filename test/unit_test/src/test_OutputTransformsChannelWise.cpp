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

#define OUTPUT_SENTINEL 99

static void set_target(void)
{
#if defined(__VX4B__)
    SetNNTargetArch(TARGET_ARCH_VX4A);
#else
    SetNNTargetArch(TARGET_ARCH_XS3A);
#endif
}

static int16_t get_multiplier()
{
    return (NN_ARCH == TARGET_ARCH_XS3A)
               ? (1 << VLMUL_SHR_XS3A)
               : ((1 << VLMUL_SHR_VX4A) - 1);
}

static int16_t pseudo_rand_int16(int *seed)
{
    return (int16_t)pseudo_rand(seed);
}

static int32_t pseudo_rand_int32(int *seed)
{
    return (int32_t)pseudo_rand(seed);
}

static int8_t channelwise_reference(int32_t accumulator, int16_t initial_shift,
                                    int16_t multiplier, int16_t bias,
                                    int16_t final_shr)
{
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A
                                                  : VLMUL_SHR_VX4A;
    int value = accumulator >> initial_shift;
    value = value < INT16_MIN ? INT16_MIN : value;
    value = value > INT16_MAX ? INT16_MAX : value;
    value = (int)(((int64_t)value * multiplier) >> sh);
    value += bias;
    value = (value + (1 << (final_shr + 7))) >> (final_shr + 8);
    return (int8_t)vpu_saturate_fixed(value, 8);
}

static int8_t *run_channelwise_test(
    int out_count, 
    int16_t final_shr,
    const int32_t *accs,
    const int16_t *init_sh,
    const int16_t *mults,
    const int16_t *biases, int8_t *out)
{
    VPURingBuffer acc{};
    int16_t params_mem[VPU_INT16_EPV * 3] = {};

    for (int ch = 0; ch < out_count; ++ch)
    {
        acc.SetAccu(ch, accs[ch]);
        params_mem[ch] = init_sh[ch];
        params_mem[out_count + ch] = mults[ch];
        params_mem[out_count * 2 + ch] = biases[ch];
    }

    nn::OT_int8_channelwise ot(out_count, final_shr);
    nn::otfn_int8_channelwise_params_t params = ot.getParams();
    return nn::otfn_int8_channelwise(&params, out, &acc, 0, params_mem);
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
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_zeros);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_sats);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_neg);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_random);
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
    const int8_t expected_unchanged = OUTPUT_SENTINEL;

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

TEST(group_output_transforms_channel_wise, Test_ot_chwise_zeros)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t multiplier = get_multiplier();
    int32_t accs[output_count] = {};
    int16_t init_sh[output_count] = {};
    int16_t mults[output_count];
    int16_t biases[output_count] = {};
    int8_t expected[output_count] = {};
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch)
    {
        mults[ch] = multiplier;
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(output_count, 0, accs, init_sh, mults, biases, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_sats)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t multiplier = get_multiplier();
    int32_t accs[output_count];
    int16_t init_sh[output_count] = {};
    int16_t mults[output_count];
    const int16_t biases[output_count] = {
        -1, -2, -3, -4, -5, -6, -7, -8,
        -9, -10, -11, -12, -13, -14, -15, -16
    };
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch)
    {
        accs[ch] = (ch & 1) ? INT16_MIN : INT16_MAX;
        mults[ch] = multiplier;
        expected[ch] = (ch & 1) ? (int8_t)vpu_saturate(-128, 8)
                     : (int8_t)vpu_saturate(127, 8);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(output_count, 0, accs, init_sh, mults, biases, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_neg)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t multiplier = get_multiplier();
    const int32_t accs[output_count] = {
        -1, -64, -127, -128, -129, -191, -255, -256,
        -257, -319, -383, -384, -385, -511, -512, -513
    };
    int16_t init_sh[output_count] = {};
    int16_t mults[output_count];
    int16_t biases[output_count] = {};
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch)
    {
        mults[ch] = multiplier;
        expected[ch] = channelwise_reference(accs[ch], 0, multiplier, biases[ch], 0);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(output_count, 0, accs, init_sh, mults, biases, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_random)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t multiplier = get_multiplier();
    int seed = 0x4F1BBCDC;
    const int16_t final_shr = (int16_t)((uint16_t)pseudo_rand_int16(&seed) % 4 + 1);
    int32_t accs[output_count];
    int16_t init_sh[output_count];
    int16_t mults[output_count];
    int16_t biases[output_count];
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch)
    {
        accs[ch] = pseudo_rand_int32(&seed);
        init_sh[ch] = (int16_t)((uint16_t)pseudo_rand_int16(&seed) % 4);
        mults[ch] = multiplier;
        biases[ch] = pseudo_rand_int16(&seed);
        expected[ch] = channelwise_reference(accs[ch], init_sh[ch], mults[ch], biases[ch], final_shr);
    }
    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(output_count, final_shr, accs, init_sh, mults, biases, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

} // extern "C"
