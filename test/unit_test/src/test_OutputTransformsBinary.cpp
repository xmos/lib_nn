// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <cstdint>

#include "OutputTransformFn.hpp"

extern "C"
{

/*
group_output_transforms_binary

Each invocation processes 16 accumulator values and packs their additive
threshold results into two output bytes:

output_bit[i] = (accumulator[i] + threshold[i] < 0) ? 1 : 0;

Test logic:
- two simple tests
- two random tests

*/

#include "etc/helpers.h"
#include "unity.h"
#include "unity_fixture.h"


TEST_GROUP(group_output_transforms_binary);
TEST_SETUP(group_output_transforms_binary) {}
TEST_TEAR_DOWN(group_output_transforms_binary) {}
TEST_GROUP_RUNNER(group_output_transforms_binary)
{
    RUN_TEST_CASE(group_output_transforms_binary, Test_otfn_binary_negative_sum);
    RUN_TEST_CASE(group_output_transforms_binary, Test_otfn_binary_positive_sum);

    RUN_TEST_CASE(group_output_transforms_binary, Test_otfn_binary_random_values);
    RUN_TEST_CASE(group_output_transforms_binary, Test_otfn_binary_large_accumulators_multiple_groups);
}

static int8_t *_test_ot_binnary_helper(
    int8_t *out,
    const int16_t accumulator,
    const int16_t threshold)
{
    // fill acc
    VPURingBuffer acc{};
    for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
        acc.SetAccu(i, accumulator);
    }
    // fill threshold array
    int16_t th[VPU_INT16_EPV];
    for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
        th[i] = threshold;
    }
    // call the output transform function
    return nn::otfn_binary(nullptr, out, &acc, 0, th);
}

TEST(group_output_transforms_binary, Test_otfn_binary_negative_sum)
{
    // a + t = -1
    const int16_t accumulator = 3;
    const int16_t threshold = -4;
    const uint8_t expected = (accumulator + threshold) < 0 ? 0xFF : 0x00;
    int8_t out[2];
    int8_t *out_end = _test_ot_binnary_helper(out, accumulator, threshold);
    TEST_ASSERT_EQUAL_PTR(out + sizeof(out), out_end);
    TEST_ASSERT_EQUAL_UINT8(expected, out[0]);
    TEST_ASSERT_EQUAL_UINT8(expected, out[1]);
}

TEST(group_output_transforms_binary, Test_otfn_binary_positive_sum)
{
    // a + t = 5
    const int16_t accumulator = 2;
    const int16_t threshold = 3;
    const uint8_t expected = (accumulator + threshold) < 0 ? 0xFF : 0x00;
    int8_t out[2];
    int8_t *out_end = _test_ot_binnary_helper(out, accumulator, threshold);
    TEST_ASSERT_EQUAL_PTR(out + sizeof(out), out_end);
    TEST_ASSERT_EQUAL_UINT8(expected, out[0]);
    TEST_ASSERT_EQUAL_UINT8(expected, out[1]);
}

TEST(group_output_transforms_binary, Test_otfn_binary_random_values)
{
    int seed = 0x6D2B79F5;
    int16_t th[VPU_INT16_EPV];
    VPURingBuffer acc{};
    int8_t out[2];

    // Generate distinct accumulator and threshold values for every lane
    for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
        const int16_t accumulator = (int16_t)pseudo_rand(&seed);
        th[i] = (int16_t)pseudo_rand(&seed);
        acc.SetAccu(i, accumulator);
    }

    // Evaluate the documented binary output condition for every lane
    bool expected[VPU_INT16_EPV];
    for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
        expected[i] = (acc.GetAccu(i) + th[i]) < 0;
    }
    int8_t *out_end = nn::otfn_binary(nullptr, out, &acc, 0, th);
    
    // Compare every expected lane against its packed output bit
    TEST_ASSERT_EQUAL_PTR(out + sizeof(out), out_end);
    for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
        const uint8_t actual = ((uint8_t)out[i / 8] >> (i % 8)) & 1;
        TEST_ASSERT_EQUAL_UINT8(expected[i], actual);
    }
}

TEST(group_output_transforms_binary,
     Test_otfn_binary_large_accumulators_multiple_groups)
{
    int seed = 0x4F1BBCDC;
    int16_t th[2 * VPU_INT16_EPV];
    VPURingBuffer acc{};
    int8_t out[2];

    // Generate full-width accumulators and two threshold groups.
    for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
        acc.SetAccu(i, pseudo_rand(&seed));
        th[i] = (int16_t)pseudo_rand(&seed);
        th[VPU_INT16_EPV + i] = (int16_t)pseudo_rand(&seed);
    }

    // Check the output for each threshold group.
    for (int32_t output_channel_group = 0;
         output_channel_group < 2;
         output_channel_group++) {
        int8_t *out_end =
            nn::otfn_binary(nullptr, out, &acc, output_channel_group, th);

        TEST_ASSERT_EQUAL_PTR(out + sizeof(out), out_end);
        for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
            const bool expected =
                ((int16_t)acc.GetAccu(i) +
                 th[output_channel_group * VPU_INT16_EPV + i]) < 0;
            const uint8_t actual = ((uint8_t)out[i / 8] >> (i % 8)) & 1;
            TEST_ASSERT_EQUAL_UINT8(expected, actual);
        }
    }
}

} // extern "C"
