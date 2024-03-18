#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "dequantize_int16.h"
#include "dequantize_int16_transform.h"

#include "tst_common.h"
#ifdef LOCAL_MAIN
    #undef UNITY_SET_FILE
int intbits(float f) {
    return *(int *)&f;
}
#define UNITY_SET_FILE()
#define RUN_TEST(x) x()
#define TEST_ASSERT_EQUAL(a, b) if ((a) != (b)) {printf("Expected %08x saw %08x\n", (int) a, (int) b); errors++;}
#define TEST_ASSERT_EQUAL_FLOAT(a, b) if ((a) != (b)) {printf("Expected %f saw %f (%08x != %08x)\n", a, b, intbits(a), intbits(b)); errors++;}
#define TEST_ASSERT_INT_WITHIN(d, a, b) if (abs((a)-(b)) > (d)) {printf("Expected %08x +/- %d saw %08x\n", (int) (a), (int) (d), (int) (b)); errors++;}
#else
#include "unity.h"
#endif

#define N 25

int test_dequantize_tensor_int16(void) {
    int16_t input1[N];
    int8_t blob[DEQUANTIZE_INT16_TENSOR_BYTES()];
    float output[N+1];
    float ref_output[N];
    int errors = 0;
    for(int i = 0; i < N; i++) {
        input1[i] = 20000 - 2513 * i;
    }
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00004051757812;
    for(int i = 0; i < N; i++) {
        float o = input1[i] * scaler1;
        ref_output[i] = o;
    }
    char err_msg[ERR_MSG_DESCRIPTOR_FAIL_BYTES()];
    int success = dequantize_int16_tensor_blob(blob,
                                               scaler1,
                                               err_msg);
    
    TEST_ASSERT_EQUAL(1, success);
   
    output[N] = 0x5555;
    dequantize_int16_tensor(output, input1, N, blob);
    TEST_ASSERT_EQUAL(output[N], 0x5555);

    for(int i = 0; i < N; i++) {
        TEST_ASSERT_EQUAL_FLOAT(ref_output[i], output[i]);
    }

    return errors;
}

void test_dequantize_int16() {
  UNITY_SET_FILE();
  RUN_TEST(test_dequantize_tensor_int16);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_dequantize_tensor_int16();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
