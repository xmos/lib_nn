// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <cmath>
#include <cstring>

#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"

extern "C"
{

#include "etc/helpers.h"
#include "unity.h"
#include "unity_fixture.h"

static int8_t expected_maxpool(
    const int8_t *input,
    unsigned channel,
    unsigned channels,
    unsigned image_width,
    unsigned kernel_width,
    unsigned kernel_height)
{
    int8_t expected = input[channel];
    for (unsigned row = 0; row < kernel_height; ++row)
    {
        for (unsigned col = 0; col < kernel_width; ++col)
        {
            const int8_t value = input[(row * image_width + col) * channels + channel];
            expected = value > expected ? value : expected;
        }
    }
    return expected;
}

TEST_GROUP(group_maxpool);
TEST_SETUP(group_maxpool) {}
TEST_TEAR_DOWN(group_maxpool) {}
TEST_GROUP_RUNNER(group_maxpool)
{
    RUN_TEST_CASE(group_maxpool, test_maxpool_simple);
    RUN_TEST_CASE(group_maxpool, test_maxpool_zeros);
    RUN_TEST_CASE(group_maxpool, test_maxpool_random);
    RUN_TEST_CASE(group_maxpool, test_maxpool_smaller_kernel);
    RUN_TEST_CASE(group_maxpool, test_output_transform_maxpool);
}

TEST(group_maxpool, test_maxpool_simple)
{
    /*
        The 2x2 input window has four values per ch:
            -100  -20
            -70   -5

        Result should be -5
    */
    const unsigned channels = VPU_INT16_EPV;
    const unsigned w = 2;
    const unsigned h = 2;

    const int8_t expected = -5;
    WORD_ALIGNED int8_t input[w * h * channels] = {};
    for (unsigned channel = 0; channel < channels; ++channel)
    {
        input[channel] = -100;
        input[channels + channel] = -20;
        input[2 * channels + channel] = -70;
        input[3 * channels + channel] = expected;
    }

    nn::ImageGeometry input_geometry(h, w, channels);
    nn::WindowGeometry kernel_geometry(h, w, 1, 1, 1, 1);
    nn::MatMulDirectFn_DW maxpool_params(input_geometry, kernel_geometry);
    nn::mat_mul_dw_direct_params_t params = maxpool_params.getParams();
    VPURingBuffer accumulator WORD_ALIGNED = {};

    nn::maxpool_direct(&params, &accumulator, input);
    for (unsigned channel = 0; channel < channels; ++channel)
    {
        TEST_ASSERT_EQUAL_INT8(expected, ((int8_t *)&accumulator.vR)[channel]);
    }
}

TEST(group_maxpool, test_maxpool_zeros)
{
    // very similar bit testing it all remains 0 for 0 inputs
    const unsigned channels = VPU_INT16_EPV;
    const unsigned w = 2;
    const unsigned h = 2;
    const int8_t expected = 0;
    WORD_ALIGNED int8_t input[w * h * channels] = {};
    VPURingBuffer accumulator WORD_ALIGNED = {};
    nn::ImageGeometry input_geometry(h, w, channels);
    nn::WindowGeometry kernel_geometry(h, w, 1, 1, 1, 1);
    nn::MatMulDirectFn_DW maxpool_params(input_geometry, kernel_geometry);
    nn::mat_mul_dw_direct_params_t params = maxpool_params.getParams();
    nn::maxpool_direct(&params, &accumulator, input);
    for (unsigned channel = 0; channel < channels; ++channel)
    {
        TEST_ASSERT_EQUAL_INT8(expected, ((int8_t *)&accumulator.vR)[channel]);
    }
}

TEST(group_maxpool, test_maxpool_random)
{
    const unsigned channels = VPU_INT16_EPV;
    const unsigned widths[] = {2, 3, 4, 5, 6, 7};
    const unsigned heights[] = {2, 10, 20, 30};
    int seed = 0x6D2B79F5;
    WORD_ALIGNED int8_t input[7 * 30 * channels]; // max size

    for (unsigned height : heights)
    {
        for (unsigned width : widths)
        {
            const unsigned pixel_count = width * height;
            for (unsigned pixel = 0; pixel < pixel_count; ++pixel)
            {
                for (unsigned channel = 0; channel < channels; ++channel)
                {
                    input[pixel * channels + channel] = (int8_t)pseudo_rand(&seed);
                }
            }

            nn::ImageGeometry input_geometry(height, width, channels);
            nn::WindowGeometry kernel_geometry(height, width, 1, 1, 1, 1);
            nn::MatMulDirectFn_DW maxpool_params(input_geometry, kernel_geometry);
            nn::mat_mul_dw_direct_params_t params = maxpool_params.getParams();
            VPURingBuffer accumulator WORD_ALIGNED = {};

            nn::maxpool_direct(&params, &accumulator, input);
            for (unsigned channel = 0; channel < channels; ++channel)
            {
                int8_t expected = expected_maxpool(
                    input, channel, channels, width, width, height);
                int8_t actual = ((int8_t *)&accumulator.vR)[channel];
                TEST_ASSERT_EQUAL_INT8(expected, actual);
            }
        }
    }
}

TEST(group_maxpool, test_maxpool_smaller_kernel)
{
    const unsigned channels = VPU_INT16_EPV;
    const unsigned image_width = 8;
    const unsigned image_height = 8;
    const unsigned kernel_width = 2;
    const unsigned kernel_height = 2;
    const unsigned start_row = 3;
    const unsigned start_col = 4;
    int seed = 0x4F1BBCDC;
    WORD_ALIGNED int8_t input[image_width * image_height * channels];

    for (unsigned pixel = 0; pixel < image_width * image_height; ++pixel)
    {
        for (unsigned channel = 0; channel < channels; ++channel)
        {
            input[pixel * channels + channel] = (int8_t)pseudo_rand(&seed);
        }
    }

    nn::ImageGeometry input_geometry(image_height, image_width, channels);
    nn::WindowGeometry kernel_geometry(kernel_height, kernel_width, 1, 1, 1, 1);
    nn::MatMulDirectFn_DW maxpool_params(input_geometry, kernel_geometry);
    nn::mat_mul_dw_direct_params_t params = maxpool_params.getParams();
    VPURingBuffer accumulator WORD_ALIGNED = {};
    int8_t *input_window = &input[(start_row * image_width + start_col) * channels];
    nn::maxpool_direct(&params, &accumulator, input_window);
    for (unsigned channel = 0; channel < channels; ++channel)
    {
        const int8_t expected = expected_maxpool(
            input_window,
            channel,
            channels,
            image_width,
            kernel_width,
            kernel_height
        );
        const int8_t actual = ((int8_t *)&accumulator.vR)[channel];
        TEST_ASSERT_EQUAL_INT8(expected, actual);
    }
}

TEST(group_maxpool, test_output_transform_maxpool)
{
    constexpr int8_t sentinel = -128;
    WORD_ALIGNED VPURingBuffer accumulator = {};
    int16_t multipliers_and_biases[VPU_INT16_EPV] = {};

    for (unsigned channel = 0; channel < VPU_INT16_EPV; ++channel)
    {
        ((int8_t *)&accumulator.vR)[channel] =
            (int8_t)(channel * 13 - 97);
    }

    for (unsigned output_count = 1; output_count <= VPU_INT16_EPV;
         ++output_count)
    {
        nn::otfn_int8_channelwise_params_t params = {
            (int32_t)output_count, 0};
        // allocating 4 bytes before the array and 1 after
        // to check that we only write in the region of interest
        // 4 bytes before to be able to word align for xs3
        WORD_ALIGNED int8_t output[VPU_INT16_EPV + 5];
        std::memset(output, sentinel, sizeof(output));

        int8_t *const end = nn::otfn_int8_maxpool(
            &params, &output[4], &accumulator, 0, multipliers_and_biases);

        TEST_ASSERT_EQUAL_PTR(&output[4 + output_count], end);
        TEST_ASSERT_EQUAL_INT8(sentinel, output[3]);
        for (unsigned channel = 0; channel < output_count; ++channel)
        {
            TEST_ASSERT_EQUAL_INT8(
                ((int8_t *)&accumulator.vR)[channel], output[4 + channel]);
        }
        TEST_ASSERT_EQUAL_INT8(sentinel, output[4 + output_count]);
    }
}

TEST(group_maxpool, Test_AvgPool2D_Global) {
  TEST_IGNORE_MESSAGE("avgpool2d_global has no implementation");
}

TEST(group_maxpool, Test_AvgPool2D_Ext) {
  TEST_IGNORE_MESSAGE("avgpool2d_ext has no implementation");
}

TEST(group_maxpool, Test_MaxPool2D_Ext) {
  TEST_IGNORE_MESSAGE("maxpool2d_ext has no implementation");
}

TEST(group_maxpool, Test_MaxPool2D_Global) {
  TEST_IGNORE_MESSAGE("maxpool2d_global has no implementation");
}

} // extern "C"
