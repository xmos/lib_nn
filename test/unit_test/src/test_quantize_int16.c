#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "quantize_int16.h"
#include "quantize_int16_transform.h"

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

int test_quantize_tensor_int16(void) {
    float input1[N];
    int8_t blob[QUANTIZE_INT16_TENSOR_BYTES()];
    int16_t output[N+1];
    int16_t ref_output[N];
    int errors = 0;
    for(int i = 0; i < N; i++) {
        input1[i] = (20000 - 2513 * i)/32768.0;
    }
    input1[3] = 2.09;
    input1[4] = -2.10;
    input1[5] = -2.11;
    float scaler1 = 0.00004051757812;
    input1[6] = -32768.5 * scaler1;
    input1[7] = -32768.0 * scaler1;
    input1[8] = -32767.5 * scaler1;
    input1[9] = 32765.5 * scaler1;
    input1[10] = 32767.0 * scaler1;
    input1[11] = 32766.5 * scaler1;
    for(int i = 0; i < N; i++) {
        float o = floor(input1[i]/scaler1 + 0.5);  // Do not round - tie goes to even
        if (o >  32767) o = 32767;
        if (o < -32768) o = -32768;
        ref_output[i] = o;
    }
    int success = quantize_int16_tensor_blob(blob,
                                             scaler1);
    
    TEST_ASSERT_EQUAL(1, success);
   
    output[N] = 0x5555;
    quantize_int16_tensor(output, input1, N, blob);
    TEST_ASSERT_EQUAL(output[N], 0x5555);

    for(int i = 0; i < N; i++) {
        TEST_ASSERT_EQUAL(ref_output[i], output[i]);
    }

    return errors;
}

void test_quantize_int16() {
  UNITY_SET_FILE();
  RUN_TEST(test_quantize_tensor_int16);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_quantize_tensor_int16();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
