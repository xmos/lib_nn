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
        y[i] = saturate_int8(round(x / 256));

    */

#include "etc/helpers.h"
#include "unity.h"
#include "unity_fixture.h"
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

TEST_GROUP(group_output_transforms_clamped);
TEST_SETUP(group_output_transforms_clamped)
{
    set_target();
}
TEST_TEAR_DOWN(group_output_transforms_clamped) {}
TEST_GROUP_RUNNER(group_output_transforms_clamped)
{
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_simple);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_simple2);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_simple3);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_random);
    RUN_TEST_CASE(group_output_transforms_clamped, Test_otfn_int8clm_random_large);
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
        expected[ch] = sat_int8(val);
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
        TEST_ASSERT_EQUAL_INT8(sat_int8(val), out[ch]);
    }
}

} // extern "C"
