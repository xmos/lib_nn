// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>
#include <math.h>

#include "nn_layers.h"
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(group_mean);
TEST_SETUP(group_mean) {}
TEST_TEAR_DOWN(group_mean) {}
TEST_GROUP_RUNNER(group_mean)
{
    RUN_TEST_CASE(group_mean, test_mean_int8);
    RUN_TEST_CASE(group_mean, test_mean_int16);
    RUN_TEST_CASE(group_mean, test_mean_random);
}

TEST(group_mean, test_mean_int8)
{
    // a dummy test yhat should be 0
    const int start_dim_size = 1;
    const int mean_dim_size = 9;
    const int end_dim_size = 1;
    const float scale_mul = 1.0f / mean_dim_size;
    const float inzp = 0.0f, outzp = 0.0f;
    const int8_t input[] = {-1, -2, -3, -4, 0, 1, 2, 3, 4};
    int8_t output[1];

    mean_int8(input, output, start_dim_size, mean_dim_size, end_dim_size, inzp, outzp, scale_mul);
    TEST_ASSERT_EQUAL_INT8(0, output[0]);
}

TEST(group_mean, test_mean_int16)
{
    // a dummy test yhat should be 0
    const int start_dim_size = 1;
    const int mean_dim_size = 9;
    const int end_dim_size = 1;
    const float scale_mul = 1.0f / mean_dim_size;
    const int16_t input[] = {-100, -200, -300, -400, 000, 100, 200, 300, 400};
    int16_t output[1];
    mean_int16(input, output, start_dim_size, mean_dim_size, end_dim_size, scale_mul);
    TEST_ASSERT_EQUAL_INT16(0, output[0]);
}

TEST(group_mean, test_mean_random)
{
    // random inputs and outputs, careful wtih clamp. %3 or %5 to make valid tensor sizes
    for (int run = 0; run < 10; run++)
    {
        const int start_dim_size = 1 + (pseudo_rand_uint32() % 3);
        const int mean_dim_size = 1 + (pseudo_rand_uint32() % 5);
        const int end_dim_size = 1 + (pseudo_rand_uint32() % 3);
        const int input_count =
            start_dim_size * mean_dim_size * end_dim_size;
        const int output_count = start_dim_size * end_dim_size;
        int8_t input8[45];
        int8_t output8[9];
        int8_t expected8[9];
        int16_t input16[45];
        int16_t output16[9];
        int16_t expected16[9];

        for (int i = 0; i < input_count; i++)
        {
            input8[i] = pseudo_rand_int8();
            input16[i] = pseudo_rand_int16();
        }

        const float int8_scale = 0.75f;
        const float int8_in_zero_point = 3.0f;
        const float int8_out_zero_point = -2.0f;
        mean_int8(input8, output8, start_dim_size, mean_dim_size,
                  end_dim_size, int8_in_zero_point, int8_out_zero_point,
                  int8_scale);

        for (int start = 0; start < start_dim_size; start++)
        {
            for (int end = 0; end < end_dim_size; end++)
            {
                int sum = 0;
                for (int mean = 0; mean < mean_dim_size; mean++)
                {
                    const int index =
                        start * mean_dim_size * end_dim_size +
                        mean * end_dim_size + end;
                    sum += input8[index];
                }
                const float value =
                    (sum - int8_in_zero_point * mean_dim_size) * int8_scale +
                    int8_out_zero_point;
                int rounded_value = (int)roundf(value);
                if (rounded_value > INT8_MAX)
                    rounded_value = INT8_MAX;
                if (rounded_value < INT8_MIN)
                    rounded_value = INT8_MIN;
                expected8[start * end_dim_size + end] = (int8_t)rounded_value;
            }
        }
        TEST_ASSERT_EQUAL_INT8_ARRAY(expected8, output8, output_count);

        const float int16_scale = 0.5f;
        mean_int16(input16, output16, start_dim_size, mean_dim_size,
                   end_dim_size, int16_scale);

        for (int start = 0; start < start_dim_size; start++)
        {
            for (int end = 0; end < end_dim_size; end++)
            {
                int32_t sum = 0;
                for (int mean = 0; mean < mean_dim_size; mean++)
                {
                    const int index =
                        start * mean_dim_size * end_dim_size +
                        mean * end_dim_size + end;
                    sum += input16[index];
                }
                int32_t rounded_value = (int32_t)roundf(sum * int16_scale);
                if (rounded_value > INT16_MAX)
                    rounded_value = INT16_MAX;
                if (rounded_value < INT16_MIN)
                    rounded_value = INT16_MIN;
                expected16[start * end_dim_size + end] = (int16_t)rounded_value;
            }
        }
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected16, output16, output_count);
    }
}
