// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "quadratic_approximation.h"
#include "quadratic_interpolation.h"

#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(group_quadratic_interpolation);
TEST_SETUP(group_quadratic_interpolation) {}
TEST_TEAR_DOWN(group_quadratic_interpolation) {}
TEST_GROUP_RUNNER(group_quadratic_interpolation) {
  RUN_TEST_CASE(group_quadratic_interpolation, test_quadratic_interpolation);
}

// Native builds can afford to sweep the full int16_t range; embedded targets
// use a smaller sweep to fit within available memory.
#if defined(TEST_BUILD_NATIVE)
#define N 65536
#else
#define N 655
#endif

TEST(group_quadratic_interpolation, test_quadratic_interpolation) {
#if defined(__VX4A__) || defined(__VX4B__)
    // KNOWN ISSUE: quadratic_interpolation_128 is not implemented on VX4 yet.
    TEST_IGNORE_MESSAGE("quadratic_interpolation_128 not implemented on VX4");
#else
    float_function_t test_functions[3] = {approximation_function_tanh,
                                          approximation_function_logistics,
                                          approximation_function_elu};
    float output_scalers[3] = {1.0/32768, 1.0/32768, 10.0/32768};
    float input_scalers[3] = {8.0/32768, 8.0/32768, 2.0/32768};

    for (int f = 0; f < 3; f++) {
        int16_t inputs[N];
        int16_t outputs[N];
        __attribute__((aligned(8))) quadratic_function_table_t table;
        uint8_t *bytes = quadratic_function_table_bytes(&table);

        double square_error;
        int max_error;
        quadratic_approximation_generator(&table, test_functions[f],
                                          input_scalers[f], output_scalers[f],
                                          128, &max_error, &square_error);

        for (int i = 0; i < N; i++) {
            inputs[i] = i * (65536 / N) - 32768;
        }
        quadratic_interpolation_128(outputs, inputs, bytes, N);

        for (int i = 0; i < N; i++) {
            float expected = (test_functions[f])(inputs[i] * input_scalers[f]) /
                             output_scalers[f];
            int err = outputs[i] - (int)roundf(expected);
            TEST_ASSERT_INT_WITHIN(1, 0, err);
        }
    }
#endif
}
