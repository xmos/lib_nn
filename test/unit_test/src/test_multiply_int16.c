#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "multiply_int16.h"
#include "multiply_int16_transform.h"

#include "tst_common.h"
#ifdef LOCAL_MAIN
    #undef UNITY_SET_FILE
#define UNITY_SET_FILE()
#define RUN_TEST(x) x()
#define TEST_ASSERT_EQUAL(a, b) if ((a) != (b)) {printf("Expected %08x saw %08x\n", (int) a, (int) b); errors++;}
#define TEST_ASSERT_INT_WITHIN(d, a, b) if (abs((a)-(b)) > (d)) {printf("Expected %08x saw %08x\n", (int) a, (int) b); errors++;}
#else
#include "unity.h"
#endif

#define N 25

int test_multiply_transform_int16(void) {
    int16_t input1[N];
    int16_t input2[N];
    int8_t transformed_input2[MULTIPLY_INT16_BYTES(N)];
    int16_t output[N+1];
    int16_t ref_output[N];
    int errors = 0;
    printf("SIZE %d\n" ,MULTIPLY_INT16_BYTES(N));
    for(int i = 0; i < N; i++) {
        input1[i] = 20000 - 2513 * i;
        input2[i] = 417 * i + 82;
    }
    input2[3] = 2;
    input2[4] = 1;
    input2[5] = -1;
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00413321;
    float scaler2 = 0.000190654;
    float scalero = 0.00318776123;
    for(int i = 0; i < N; i++) {
        float oo = input1[i] * input2[i] * scaler1 * scaler2 / scalero;
        float o = round(oo);
        if (o >  32767) o =  32767;
        if (o < -32768) o = -32768;
        ref_output[i] = o;
    }
    multiply_int16_transform(transformed_input2,
                             input2,
                             N,
                             scaler1,
                             scaler2,
                             scalero);
    output[N] = 0x5555;
    multiply_int16_elementwise_constant(output, input1, transformed_input2, N);
    TEST_ASSERT_EQUAL(output[N], 0x5555);

    for(int i = 0; i < N; i++) {
        printf("%d: %d %d\n", i, output[i], ref_output[i]);
        TEST_ASSERT_INT_WITHIN(1, ref_output[i], output[i]);
    }

    return errors;
}

int test_requantise_transform_int16(void) {
    int16_t input1[N];
    int8_t requantise_blob[REQUANTISE_INT16_BYTES(N)];
    int16_t output[N+1];
    int16_t req_output[N];
    int errors = 0;
    printf("SIZE %d\n" ,REQUANTISE_INT16_BYTES(N));
    for(int i = 0; i < N; i++) {
        input1[i] = 20000 - 2513 * i;
    }
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00413321;
    float scalero = 0.00318776123;
    for(int i = 0; i < N; i++) {
        float oo = input1[i] * scaler1 / scalero;
        float o = round(oo);
        if (o >  32767) o =  32767;
        if (o < -32768) o = -32768;
        req_output[i] = o;
    }

    requantise_int16_transform(requantise_blob,
                               N,
                               scaler1,
                               scalero);
    output[N] = 0x5555;
    multiply_int16_elementwise_constant(output, input1, requantise_blob, N);
    TEST_ASSERT_EQUAL(output[N], 0x5555);

    for(int i = 0; i < N; i++) {
        printf("%d: %d %d\n", i, output[i], req_output[i]);
        TEST_ASSERT_INT_WITHIN(1, req_output[i], output[i]);
    }
    return errors;
}


void test_multiply_int16() {
  UNITY_SET_FILE();
  RUN_TEST(test_multiply_transform_int16);
  RUN_TEST(test_requantise_transform_int16);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_multiply_transform_int16();
    errors += test_requantise_transform_int16();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
