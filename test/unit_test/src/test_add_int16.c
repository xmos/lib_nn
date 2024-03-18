#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "add_int16.h"
#include "add_int16_transform.h"

#include "tst_common.h"
#ifdef LOCAL_MAIN
    #undef UNITY_SET_FILE
#define UNITY_SET_FILE()
#define RUN_TEST(x) x()
#define TEST_ASSERT_EQUAL(a, b) if ((a) != (b)) {printf("Expected %08x saw %08x\n", (int) a, (int) b); errors++;}
#define TEST_ASSERT_INT_WITHIN(d, a, b) if (abs((a)-(b)) > (d)) {printf("Expected %08x +/- %d saw %08x\n", (int) (a), (int) (d), (int) (b)); errors++;}
#else
#include "unity.h"
#endif

#define N 39

int test_add_tensor_int16(void) {
    int16_t input1[N];
    int16_t input2[N];
    int8_t blob[ADD_INT16_TENSOR_BYTES()];
    int16_t output[N+1];
    int16_t ref_output[N];
    int errors = 0;
    for(int j=1; j < N; j++) {
    for(int i = 0; i < j; i++) {
        input1[i] = 20000 - 2513 * i;
        input2[i] = 417 * i + 82;
    }
    input2[3] = 30001;
    input2[4] = 31003;
    input2[5] = -32003;
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00004051757812;
    float scaler2 = 0.00006123190654;
    float scalero = 0.00006213;
    for(int i = 0; i < j; i++) {
        float oo = (input1[i] * scaler1 + input2[i] * scaler2) / scalero;
        float o = round(oo);
        if (o >  32767) o =  32767;
        if (o < -32768) o = -32768;
        ref_output[i] = o;
    }
    char err_msg[ERR_MSG_DESCRIPTOR_FAIL_BYTES()];
    int success = add_int16_tensor_blob(blob,
                                        scaler1,
                                        scaler2,
                                        scalero,
                                        err_msg);
    
    TEST_ASSERT_EQUAL(1, success);
   
    output[j] = 0x5555;
    add_int16_tensor(output, input1, input2, j, blob);
    TEST_ASSERT_EQUAL(output[j], 0x5555);

    int sqerr = 0;
    for(int i = 0; i < j; i++) {
        int err = ref_output[i] - output[i];
   //     printf("%04x %04x %d\n", ref_output[i] & 0xffff, output[i] & 0xffff, err);
        sqerr += err*err;
        TEST_ASSERT_INT_WITHIN(1, ref_output[i], output[i]);
    }
    TEST_ASSERT_INT_WITHIN(8, sqerr, 0);
    }
    return errors;
}

void test_add_int16() {
  UNITY_SET_FILE();
  RUN_TEST(test_add_tensor_int16);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_add_tensor_int16();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
