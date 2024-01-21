#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "expand_8_to_16.h"

#include "tst_common.h"
#ifdef LOCAL_MAIN
    #undef UNITY_SET_FILE
#define UNITY_SET_FILE()
#define RUN_TEST(x) x()
#define TEST_ASSERT_EQUAL(a, b)   if ((a) != (b)) {printf("Expected %08x saw %08x\n", (int) a, (int) b); errors++;}
#else
#include "unity.h"
#endif

int8_t inputs[64];
int16_t outputs[72];

int Test_expand_8_to_16() {
    int errors = 0;
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
    return errors;
}

void test_expand_8_to_16() {
  UNITY_SET_FILE();
  RUN_TEST(Test_expand_8_to_16);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += Test_expand_8_to_16();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
