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
    int value = accumulator;
    if (initial_shift > 0)
    {
        value += 1 << (initial_shift - 1);
    }
    value >>= initial_shift;
    value = (int)vpu_saturate(value, 16);
    value = (int)(((int64_t)value * multiplier + (1LL << (sh - 1))) >> sh);
    value = (int)vpu_saturate(value, 16);
    value = (int)vpu_saturate(value + bias, 16);
    value = (value + (1 << (final_shr + 7))) >> (final_shr + 8);
    return saturate_output_int8(value);
}

static int8_t *run_channelwise_test(
    int out_count,
    int16_t final_shr,
    const int32_t *accs,
    const int16_t *init_sh,
    const int16_t *mults,
    const int16_t *biases,
    int32_t out_chgroup,
    int8_t *out)
{
    VPURingBuffer acc{};
    int16_t params_mem[VPU_INT16_EPV * 6] = {};
    const int group_offset = out_chgroup * VPU_INT16_EPV;
    const int channels_in_group =
        (out_count - group_offset < VPU_INT16_EPV)
            ? out_count - group_offset
            : VPU_INT16_EPV;

    for (int ch = 0; ch < channels_in_group; ++ch)
    {
        acc.SetAccu(ch, accs[group_offset + ch]);
    }

    for (int group = 0; group * VPU_INT16_EPV < out_count; ++group)
    {
        const int start = group * VPU_INT16_EPV;
        const int count = (out_count - start < VPU_INT16_EPV)
                              ? out_count - start
                              : VPU_INT16_EPV;
        int16_t *group_params = &params_mem[group * VPU_INT16_EPV * 3];

        for (int ch = 0; ch < count; ++ch)
        {
            group_params[ch] = init_sh[start + ch];
            group_params[count + ch] = mults[start + ch];
            group_params[count * 2 + ch] = biases[start + ch];
        }
    }

    nn::OT_int8_channelwise ot(out_count, final_shr);
    nn::otfn_int8_channelwise_params_t params = ot.getParams();
    return nn::otfn_int8_channelwise(&params, out, &acc, out_chgroup, params_mem);
}

TEST_GROUP(group_output_transforms_channel_wise);
TEST_SETUP(group_output_transforms_channel_wise) {}
TEST_TEAR_DOWN(group_output_transforms_channel_wise) {}
TEST_GROUP_RUNNER(group_output_transforms_channel_wise)
{
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_simple);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_zeros);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_sats);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_neg);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_random);
    RUN_TEST_CASE(group_output_transforms_channel_wise, Test_ot_chwise_multiple_groups);
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
    const int out_count = 2;
    const int32_t out_chgroup = 0; 
    const int16_t final_shr = 0;
    const int16_t multiplier = get_multiplier();
    const int8_t expected[out_count] = {1, 2};
    const int8_t expected_unchanged = OUTPUT_SENTINEL;

    VPURingBuffer acc{};
    acc.SetAccu(0, 256);
    acc.SetAccu(1, 512);

    // The channelwise layout is [initial shifts][multipliers][biases].
    int16_t params_mem[out_count * 3] = {0, 0, multiplier, multiplier, 0, 0};
    int8_t out[out_count + 1] = {0, 0, expected_unchanged};

    nn::OT_int8_channelwise ot(out_count, final_shr);
    nn::otfn_int8_channelwise_params_t params = ot.getParams();
    int8_t *end = nn::otfn_int8_channelwise(&params, out, &acc, out_chgroup, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + out_count, end);
    for (int ch = 0; ch < out_count; ++ch)
    {
        TEST_ASSERT_EQUAL_INT8(expected[ch], out[ch]);
    }
    TEST_ASSERT_EQUAL_INT8(expected_unchanged, out[out_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_zeros)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t final_shr = 0;
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
    int8_t *end = run_channelwise_test(
        output_count, final_shr, accs, init_sh, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_sats)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t multiplier = get_multiplier();
    const int16_t final_shr = 0;
    const int16_t boundary_bias =
        (NN_ARCH == TARGET_ARCH_XS3A) ? 0 : -1;
    const int8_t int8_max = (int8_t)vpu_saturate_fixed(INT8_MAX, 8);
    const int8_t int8_min = INT8_MIN; //Note: vdepth8 is corrected in xs3
    const int32_t accs[output_count] = {
        INT16_MAX, -32640, INT16_MIN, -32641,
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN
    };
    int16_t init_sh[output_count] = {};
    int16_t mults[output_count];
    const int16_t biases[output_count] = {
        0, boundary_bias, 0, boundary_bias, -5, -6, -7, -8,
        -9, -10, -11, -12, -13, -14, -15, -16
    };
    const int8_t expected[output_count] = {
        int8_max, -127, int8_min, int8_min, int8_max, int8_min, int8_max, int8_min,
        int8_max, int8_min, int8_max, int8_min, int8_max, int8_min, int8_max, int8_min
    };
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch)
    {
        mults[ch] = multiplier;
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(
        output_count, final_shr, accs, init_sh, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_neg)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t final_shr = 0;
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
        expected[ch] = channelwise_reference(accs[ch], final_shr, multiplier, biases[ch], 0);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(
        output_count, final_shr, accs, init_sh, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_random)
{
    const int output_count = VPU_INT16_EPV;
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
        mults[ch] = pseudo_rand_int16(&seed);
        biases[ch] = pseudo_rand_int16(&seed);
        expected[ch] = channelwise_reference(accs[ch], init_sh[ch], mults[ch], biases[ch], final_shr);
    }
    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(
        output_count, final_shr, accs, init_sh, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_channel_wise, Test_ot_chwise_multiple_groups)
{
    const int output_count = VPU_INT16_EPV + 4;
    int32_t accs[output_count];
    int16_t init_sh[output_count] = {};
    int16_t mults[output_count];
    int16_t biases[output_count];
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch)
    {
        accs[ch] = 256 * (ch + 1);
        mults[ch] = (ch < VPU_INT16_EPV) ? get_multiplier()
                                          : (get_multiplier() >> 1);
        biases[ch] = (ch < VPU_INT16_EPV) ? 0 : 256;
        expected[ch] =
            channelwise_reference(accs[ch], init_sh[ch], mults[ch], biases[ch], 0);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_channelwise_test(
        output_count, 0, accs, init_sh, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + VPU_INT16_EPV, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, VPU_INT16_EPV);

    end = run_channelwise_test(
        output_count, 0, accs, init_sh, mults, biases, 1, end);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

} // extern "C"
