// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <algorithm>
#include <cstdint>

#include "OutputTransformFn.hpp"

extern "C"
{

    /*
    group_output_transforms_clamped


    int8_t *output_transform_fn_int_clamped_ref(
        const otfn_int8_clamped_params_t *params,
        int8_t *Y,
        VPURingBuffer *A,
        int32_t output_channel_group,
        int16_t *offsets_multipliers_and_biases
    )

    For channel i, the operation is approximately:
        x = accumulator[i] + offset[i];
        x = max(x, 0);
        x = shift(x, initial_shift);
        x = fixed_point_multiply(x, multiplier[i]);
        x = x + bias[i];
        x = shift(x, final_shr);
        y[i] = saturate_output_int8(round(x / 256));

    */

#include "etc/helpers.h"
#include "unity.h"
#include "unity_fixture.h"
#include "vpu_sim.h"

static int16_t get_multiplier(void)
{
    return (NN_ARCH == TARGET_ARCH_XS3A)
               ? (1 << VLMUL_SHR_XS3A)
               : ((1 << VLMUL_SHR_VX4A) - 1);
}

static int8_t clamped_reference(int32_t accumulator, int16_t offset,
                                int16_t initial_shift, int16_t multiplier,
                                int16_t bias, int16_t final_shr)
{
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A
                                                  : VLMUL_SHR_VX4A;
    int value = (int)vpu_saturate(accumulator + offset, 16);
    value = value > 0 ? value : 0;
    if (initial_shift > 0) {
        value += 1 << (initial_shift - 1);
    }
    value >>= initial_shift;
    value = (int)(((int64_t)value * multiplier + (1LL << (sh - 1))) >> sh);
    value = (int)vpu_saturate(value, 16);
    value = (int)vpu_saturate(value + bias, 16);
    value = (value + (1 << (final_shr + 7))) >> (final_shr + 8);
    return saturate_output_int8(value);
}

static int8_t *run_clamped_test(
    int out_count, int16_t initial_shift, int16_t final_shr,
    const int32_t *accumulators, const int16_t *offsets,
    const int16_t *multipliers, const int16_t *biases,
    int32_t output_channel_group, int8_t *out)
{
    VPURingBuffer acc{};
    int16_t params_mem[VPU_INT16_EPV * 6] = {};
    const int group_offset = output_channel_group * VPU_INT16_EPV;
    const int channels_in_group =
        (out_count - group_offset < VPU_INT16_EPV)
            ? out_count - group_offset
            : VPU_INT16_EPV;

    for (int ch = 0; ch < channels_in_group; ++ch) {
        acc.SetAccu(ch, accumulators[group_offset + ch]);
    }
    for (int group = 0; group * VPU_INT16_EPV < out_count; ++group) {
        const int start = group * VPU_INT16_EPV;
        const int count = (out_count - start < VPU_INT16_EPV)
                              ? out_count - start
                              : VPU_INT16_EPV;
        int16_t *group_params = &params_mem[group * VPU_INT16_EPV * 3];

        for (int ch = 0; ch < count; ++ch) {
            group_params[ch] = offsets[start + ch];
            group_params[count + ch] = multipliers[start + ch];
            group_params[count * 2 + ch] = biases[start + ch];
        }
    }

    nn::OT_int8_clamped ot(out_count, initial_shift, final_shr);
    nn::otfn_int8_clamped_params_t params = ot.getParams();
    return nn::otfn_int8_clamped(&params, out, &acc, output_channel_group,
                                  params_mem);
}

TEST_GROUP(group_output_transforms_clamped);
TEST_SETUP(group_output_transforms_clamped) {}
TEST_TEAR_DOWN(group_output_transforms_clamped) {}
TEST_GROUP_RUNNER(group_output_transforms_clamped)
{
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_simple);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_simple2);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_simple3);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_random);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_random_large);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_corner_cases);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_multiple_groups);
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_simple)
{
    /*
    the idea here is
        accumulator + offset = 500 + 12 = 512
        initial shift        = 512 >> 1 = 256
        VLMUL                = 256 * shift / multiplier = 256
        final shift          = 256 >> 0 = 256
        VDEPTH8              = 256 / 256 = 1
        y = 1
    */
    const int oc = VPU_INT16_EPV;
    const int16_t acc_val = 500;
    const int16_t off = 12;
    const int16_t ish = 1;
    const int16_t bia = 0;
    const int16_t fshr = 0;
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
    // VX4 unity is Q15; 1 << 15 does not fit in int16_t.
    const int16_t mul = sh == 15 ? INT16_MAX : (1 << sh);
    
    const int8_t expected = 1;

    VPURingBuffer acc{};
    acc.SetAccu(0, acc_val);

    int8_t out[oc] = {0};
    int16_t params_mem[oc * 3] = {0};
    params_mem[0] = off;
    params_mem[oc] = mul;
    params_mem[oc * 2] = bia;

    nn::OT_int8_clamped ot(oc, ish, fshr);
    nn::otfn_int8_clamped_params_t params = ot.getParams();
    int8_t *end = nn::otfn_int8_clamped(&params, out, &acc, 0, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + oc, end);
    TEST_ASSERT_EQUAL_INT8(expected, out[0]);
    
    // the rest should be zeros (no memory overwritten)
    for (int ch = 1; ch < oc; ++ch)
    {
        TEST_ASSERT_EQUAL_INT8(0, out[ch]);
    }
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_simple2)
{
    /*
    the idea here is
        accumulator + offset = 16372 + 12 = 16384
        initial shift        = 16384 >> 1 = 8192
        VLMUL                = 8192 * 0.5 = 4096
        final shift          = 4096
        VDEPTH8              = 4096 / 256 = 16
        y = 16
    */
    const int oc = VPU_INT16_EPV;
    const int16_t acc_val = 16372;
    const int16_t off = 12;
    const int16_t ish = 1;
    const int16_t bia = 0;
    const int16_t fshr = 0;
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
    const int16_t mul = 1 << (sh - 1);
    const int8_t expected = 16;

    VPURingBuffer acc{};
    acc.SetAccu(0, acc_val);

    int8_t out[oc] = {0};
    int16_t params_mem[oc * 3] = {0};
    params_mem[0] = off;
    params_mem[oc] = mul;
    params_mem[oc * 2] = bia;

    nn::OT_int8_clamped ot(oc, ish, fshr);
    nn::otfn_int8_clamped_params_t params = ot.getParams();
    int8_t *end = nn::otfn_int8_clamped(&params, out, &acc, 0, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + oc, end);
    TEST_ASSERT_EQUAL_INT8(expected, out[0]);

    for (int ch = 1; ch < oc; ++ch)
    {
        TEST_ASSERT_EQUAL_INT8(0, out[ch]);
    }
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_simple3)
{
    const int oc = VPU_INT16_EPV;
    const int16_t acc_val = 472;
    const int16_t off = 29;
    const int16_t ish = 1;
    const int16_t bia = 76;
    const int16_t fshr = 0;
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
    const int16_t mul = sh == 15 ? INT16_MAX : (1 << sh);
    const int8_t expected = 1;

    VPURingBuffer acc{};
    acc.SetAccu(0, acc_val);

    int8_t out[oc] = {0};
    int16_t params_mem[oc * 3] = {0};
    params_mem[0] = off;
    params_mem[oc] = mul;
    params_mem[oc * 2] = bia;

    nn::OT_int8_clamped ot(oc, ish, fshr);
    nn::otfn_int8_clamped_params_t params = ot.getParams();
    int8_t *end = nn::otfn_int8_clamped(&params, out, &acc, 0, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + oc, end);
    TEST_ASSERT_EQUAL_INT8(expected, out[0]);

    for (int ch = 1; ch < oc; ++ch)
    {
        TEST_ASSERT_EQUAL_INT8(0, out[ch]);
    }
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_random)
{
    const int oc = VPU_INT16_EPV;
    const int16_t ish = 1;
    const int16_t fshr = 0;
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
    const int16_t mul = sh == 15 ? INT16_MAX : (1 << sh);
    int seed = 0x6D2B79F5;

    VPURingBuffer acc{};
    int16_t params_mem[oc * 3] = {};
    int expected[oc] = {};

    for (int ch = 0; ch < oc; ++ch)
    {
        const int16_t aval = (int16_t)((uint32_t)pseudo_rand(&seed) % 2001);
        const int16_t off = (int16_t)((uint32_t)pseudo_rand(&seed) % 101);
        const int16_t bia = (int16_t)((uint32_t)pseudo_rand(&seed) % 101);
        acc.SetAccu(ch, aval);
        params_mem[ch] = off;
        params_mem[oc + ch] = mul;
        params_mem[oc * 2 + ch] = bia;

        int val = (int)aval + off;
        val = val > 0 ? val : 0;
        val = (val + (1 << (ish - 1))) >> ish;
        val += bia;
        val = (val + 128) >> 8;
        expected[ch] = saturate_output_int8(val);
    }

    nn::OT_int8_clamped ot(oc, ish, fshr);
    nn::otfn_int8_clamped_params_t params = ot.getParams();
    int8_t out[oc] = {};
    int8_t *end = nn::otfn_int8_clamped(&params, out, &acc, 0, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + oc, end);
    for (int ch = 0; ch < oc; ++ch)
    {
        TEST_ASSERT_EQUAL_INT8(expected[ch], out[ch]);
    }
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_random_large)
{
    const int oc = VPU_INT16_EPV;
    const int16_t ish = 0;
    const int16_t off = 0;
    const int16_t bia = 0;
    const int16_t fshr = 0;
    const int sh = (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
    const int16_t mul = sh == 15 ? INT16_MAX : (1 << sh);
    int seed = 0x4F1BBCDC;

    VPURingBuffer acc{};
    int16_t params_mem[oc * 3] = {};
    for (int ch = 0; ch < oc; ++ch)
    {
        const int16_t aval = (int16_t)((int)((uint32_t)pseudo_rand(&seed) % 60001) - 30000);
        acc.SetAccu(ch, aval);
        params_mem[ch] = off;
        params_mem[oc + ch] = mul;
        params_mem[oc * 2 + ch] = bia;
    }

    nn::OT_int8_clamped ot(oc, ish, fshr);
    nn::otfn_int8_clamped_params_t params = ot.getParams();
    int8_t out[oc] = {};
    int8_t *end = nn::otfn_int8_clamped(&params, out, &acc, 0, params_mem);

    TEST_ASSERT_EQUAL_PTR(out + oc, end);
    for (int ch = 0; ch < oc; ++ch)
    {
        int val = (int)acc.GetAccu(ch);
        val = val > 0 ? val : 0;
        val = (val + 128) >> 8;
        TEST_ASSERT_EQUAL_INT8(saturate_output_int8(val), out[ch]);
    }
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_corner_cases)
{
    const int output_count = VPU_INT16_EPV;
    const int16_t initial_shift = 3;
    const int16_t final_shr = 2;
    const int16_t unity = get_multiplier();
    const int16_t half_unity = (int16_t)(unity >> 1);
    const int16_t negative_half_unity = (int16_t)-half_unity;
    const int32_t accumulators[output_count] = {
        INT16_MIN, -257, -1, 0, 1, 127, 128, 255,
        256, 1023, 1024, INT16_MAX, INT16_MAX, -INT16_MAX, INT16_MAX, -512};
    const int16_t offsets[output_count] = {
        0, 256, 1, 0, -1, -128, -129, 1,
        -256, 0, 1, 1, -1, INT16_MAX, 0, 512};
    const int16_t multipliers[output_count] = {
        unity, half_unity, negative_half_unity, 0,
        unity, half_unity, negative_half_unity, 0,
        unity, half_unity, negative_half_unity, 0,
        unity, half_unity, (int16_t)-unity, 0};
    const int16_t biases[output_count] = {
        0, 1, -1, INT16_MAX, -INT16_MAX, 127, -128, 256,
        -256, 1024, -1024, 0, 1, -1, -1000, -327};
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        expected[ch] = clamped_reference(accumulators[ch], offsets[ch],
                                         initial_shift, multipliers[ch],
                                         biases[ch], final_shr);
    }

    out[output_count] = 99;
    int8_t *end = run_clamped_test(
        output_count, initial_shift, final_shr, accumulators, offsets,
        multipliers, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(99, out[output_count]);
}

TEST(group_output_transforms_clamped, Test_otfn_int8clm_multiple_groups)
{
    const int output_count = VPU_INT16_EPV + 4;
    int32_t accumulators[output_count];
    int16_t offsets[output_count];
    int16_t multipliers[output_count];
    int16_t biases[output_count];
    int8_t expected[output_count];
    int8_t out[output_count + 1] = {};

    for (int ch = 0; ch < output_count; ++ch) {
        accumulators[ch] = 256 * (ch + 1);
        offsets[ch] = (ch < VPU_INT16_EPV) ? 0 : -128;
        multipliers[ch] = (ch < VPU_INT16_EPV) ? get_multiplier()
                                                : (get_multiplier() >> 1);
        biases[ch] = (ch < VPU_INT16_EPV) ? 0 : 256;
        expected[ch] = clamped_reference(
            accumulators[ch], offsets[ch], 1, multipliers[ch], biases[ch], 1);
    }

    out[output_count] = 99;
    int8_t *end = run_clamped_test(
        output_count, 1, 1, accumulators, offsets, multipliers, biases, 0, out);
    TEST_ASSERT_EQUAL_PTR(out + VPU_INT16_EPV, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, VPU_INT16_EPV);

    end = run_clamped_test(
        output_count, 1, 1, accumulators, offsets, multipliers, biases, 1, end);
    TEST_ASSERT_EQUAL_PTR(out + output_count, end);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, output_count);
    TEST_ASSERT_EQUAL_INT8(99, out[output_count]);
}

} // extern "C"
