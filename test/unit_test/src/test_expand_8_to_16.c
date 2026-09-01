// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "expand_8_to_16.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

int8_t inputs[64];
int16_t outputs[72];

TEST_GROUP(group_expand_8_to_16);
TEST_SETUP(group_expand_8_to_16) {}
TEST_TEAR_DOWN(group_expand_8_to_16) {}
TEST_GROUP_RUNNER(group_expand_8_to_16) {
    RUN_TEST_CASE(group_expand_8_to_16, Test_expand_8_to_16);
}

TEST(group_expand_8_to_16, Test_expand_8_to_16) {

    for(int i = 0; i < 64; i++) {
        inputs[i] = (int8_t)(i*i);
    }
    for(int j = 0; j < 64; j++) {
        for(int i = 0; i < 72; i++) {
            outputs[i] = (int16_t)(i^0xFFFF);
        }
        expand_8_to_16(outputs+4, inputs, j);
        for(int i = 0; i < 72; i++) {
            if (i < 4 || i >= 68) {
                TEST_ASSERT_EQUAL(outputs[i], (int16_t)(i^0xFFFF));
            }
        }
        for(int i = 0; i < j; i++) {
            TEST_ASSERT_EQUAL(outputs[i+4], inputs[i]);
        }
    }
}
