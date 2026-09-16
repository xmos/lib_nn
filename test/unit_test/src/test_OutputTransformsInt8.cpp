// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <cstdint>

#include "OutputTransformFn.hpp"

extern "C" {
#include "etc/helpers.h"
#include "unity.h"
#include "unity_fixture.h"
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

static int16_t get_multiplier(void)
{
    return (NN_ARCH == TARGET_ARCH_XS3A)
               ? (1 << VLMUL_SHR_XS3A)
               : ((1 << VLMUL_SHR_VX4A) - 1);
}

static int8_t groupwise_reference(int32_t accumulator, int16_t initial_shift,
                                  int16_t multiplier, int16_t bias,
                                  int16_t final_shr)
{
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A
                                                  : VLMUL_SHR_VX4A;
    int value = accumulator;

    if (initial_shift > 0) {
        value += 1 << (initial_shift - 1);
        value >>= initial_shift;
    } else if (initial_shift < 0) {
        value <<= -initial_shift;
    }

    value = (int)vpu_saturate(value, 16);
    value = (int)(((int64_t)value * multiplier + (1LL << (sh - 1))) >> sh);
    value = (int)vpu_saturate(value, 16);
    value = (int)vpu_saturate(value + bias, 16);
    value = (value + (1 << (final_shr + 7))) >> (final_shr + 8);
    return (int8_t)vpu_saturate_fixed(value, 8);
}

static int8_t *run_groupwise_test(int out_count, int16_t initial_shift,
                                  int16_t final_shr, const int32_t *accs,
                                  const int16_t *mults, const int16_t *biases,
                                  int32_t output_channel_group, int8_t *out)
{
    VPURingBuffer acc{};
    int16_t params_mem[VPU_INT16_EPV * 4] = {};
    const int group_offset = output_channel_group * VPU_INT16_EPV;
    const int channels_in_group =
        (out_count - group_offset < VPU_INT16_EPV)
            ? out_count - group_offset
            : VPU_INT16_EPV;

    for (int ch = 0; ch < channels_in_group; ++ch) {
        acc.SetAccu(ch, accs[group_offset + ch]);
    }

    for (int group = 0; group * VPU_INT16_EPV < out_count; ++group) {
        const int start = group * VPU_INT16_EPV;
        const int count = (out_count - start < VPU_INT16_EPV)
                              ? out_count - start
                              : VPU_INT16_EPV;
        int16_t *group_params = &params_mem[group * VPU_INT16_EPV * 2];

        for (int ch = 0; ch < count; ++ch) {
            group_params[ch] = mults[start + ch];
            group_params[count + ch] = biases[start + ch];
        }
    }

    nn::OT_int8 ot(out_count, initial_shift, final_shr);
    nn::otfn_int8_params_t params = ot.getParams();
    return nn::otfn_int8(&params, out, &acc, output_channel_group, params_mem);
}

TEST_GROUP(group_output_transforms_int8);
TEST_SETUP(group_output_transforms_int8)
{
    set_target();
}
TEST_TEAR_DOWN(group_output_transforms_int8) {}
TEST_GROUP_RUNNER(group_output_transforms_int8)
{
    RUN_TEST_CASE(group_output_transforms_int8, Test_ot_int8_simple);
    RUN_TEST_CASE(group_output_transforms_int8, Test_ot_int8_zeros);
    RUN_TEST_CASE(group_output_transforms_int8, Test_ot_int8_sats);
    RUN_TEST_CASE(group_output_transforms_int8, Test_ot_int8_neg);
    RUN_TEST_CASE(group_output_transforms_int8, Test_ot_int8_random);
    RUN_TEST_CASE(group_output_transforms_int8, Test_ot_int8_multiple_groups);
}

TEST(group_output_transforms_int8, Test_ot_int8_simple)
{
    const int out_count = 2;
    const int16_t multiplier = get_multiplier();
    const int32_t accs[out_count] = {256, 512};
    const int16_t mults[out_count] = {multiplier, multiplier};
    const int16_t biases[out_count] = {0, 0};
    const int8_t expected[out_count] = {1, 2};
    int8_t out[out_count + 1] = {0, 0, OUTPUT_SENTINEL};

    int8_t *end =
        run_groupwise_test(out_count, 0, 0, accs, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + out_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, out_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[out_count]);
}

TEST(group_output_transforms_int8, Test_ot_int8_zeros)
{
    const int output_count = VPU_INT16_EPV;
    int32_t accs[output_count] = {};
    int16_t mults[output_count];
    int16_t biases[output_count] = {};
    int8_t expected[output_count] = {};
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        mults[ch] = get_multiplier();
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end =
        run_groupwise_test(output_count, 0, 0, accs, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_int8, Test_ot_int8_sats)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t multiplier = get_multiplier();
    const int32_t accs[output_count] = {
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN,
        INT16_MAX, INT16_MIN, INT16_MAX, INT16_MIN};
    int16_t mults[output_count];
    const int16_t biases[output_count] = {
        -1, -2, -3, -4, -5, -6, -7, -8,
        -9, -10, -11, -12, -13, -14, -15, -16};
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        mults[ch] = multiplier;
        expected[ch] =
            groupwise_reference(accs[ch], 0, mults[ch], biases[ch], 0);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end =
        run_groupwise_test(output_count, 0, 0, accs, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_int8, Test_ot_int8_neg)
{
    const int output_count = VPU_INT16_EPV;
    const int32_t accs[output_count] = {
        -1, -64, -127, -128, -129, -191, -255, -256,
        -257, -319, -383, -384, -385, -511, -512, -513};
    int16_t mults[output_count];
    int16_t biases[output_count] = {};
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        mults[ch] = get_multiplier();
        expected[ch] = groupwise_reference(accs[ch], 0, mults[ch], biases[ch], 0);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end =
        run_groupwise_test(output_count, 0, 0, accs, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_int8, Test_ot_int8_random)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t initial_shift = 3;
    const int16_t final_shr = 2;
    int seed = 0x4F1BBCDC;
    int32_t accs[output_count];
    int16_t mults[output_count];
    int16_t biases[output_count];
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        accs[ch] = pseudo_rand(&seed);
        mults[ch] = (int16_t)pseudo_rand(&seed);
        biases[ch] = (int16_t)pseudo_rand(&seed);
        expected[ch] = groupwise_reference(
            accs[ch], initial_shift, mults[ch], biases[ch], final_shr);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end = run_groupwise_test(output_count, initial_shift, final_shr,
                                     accs, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

TEST(group_output_transforms_int8, Test_ot_int8_multiple_groups)
{
    const int output_count = VPU_INT16_EPV + 3;
    int32_t accs[output_count];
    int16_t mults[output_count];
    int16_t biases[output_count];
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        accs[ch] = 256 * (ch + 1);
        mults[ch] = (ch < VPU_INT16_EPV) ? get_multiplier()
                                          : (get_multiplier() >> 1);
        biases[ch] = (ch < VPU_INT16_EPV) ? 0 : 256;
        expected[ch] = groupwise_reference(accs[ch], 0, mults[ch], biases[ch], 0);
    }

    out[output_count] = OUTPUT_SENTINEL;
    int8_t *end =
        run_groupwise_test(output_count, 0, 0, accs, mults, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + VPU_INT16_EPV, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, VPU_INT16_EPV);

    end = run_groupwise_test(output_count, 0, 0, accs, mults, biases, 1, end);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(OUTPUT_SENTINEL, out[output_count]);
}

} // extern "C"
