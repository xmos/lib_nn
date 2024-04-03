#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "vpu_memset_256.h"

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

int test_vpu_memset256(void) {
    int mem1[32];
    int from[4][8] =
        {
            {0x80706050, 0x80706050, 0x80706050, 0x80706050,
             0x80706050, 0x80706050, 0x80706050, 0x80706050}, // int32
            {0x80808080, 0x80808080, 0x80808080, 0x80808080,
             0x80808080, 0x80808080, 0x80808080, 0x80808080}, // int8
            {0x80708070, 0x80708070, 0x80708070, 0x80708070,
             0x80708070, 0x80708070, 0x80708070, 0x80708070}, // int16
            {0x80808080, 0x80808080, 0x80808080, 0x80808080,
             0x80808080, 0x80808080, 0x80808080, 0x80808080}, // int8
        };
    int dst_vals[] = {1, 2, 3, 4, 5};
    int len_vals[] = {0, 1, 2, 3, 4, 5, 16, 30, 31, 32, 33, 34, 35, 36, 37, 38, 59, 60, 61, 62, 63};
    int errors = 0;
    int time = 0;
    for(int len_index = 0; len_index < sizeof(len_vals)/sizeof(int); len_index++) {
        int len = len_vals[len_index];
        for(int dst_index = 0; dst_index < sizeof(dst_vals)/sizeof(int); dst_index++) {
            unsigned t0, t1;
            int dst = dst_vals[dst_index];
//            printf("Dst %d  len %d  \n", dst, len);
            for(int k = 0; k < 32; k++) {
                mem1[k] = 0;
            }
            asm volatile("gettime %0" : "=r" (t0));
            vpu_memset_256(((uint8_t*)mem1)+dst, ((uint8_t*)(from[dst&3])), len);
            asm volatile("gettime %0" : "=r" (t1));
            time += t1-t0;
            TEST_ASSERT_EQUAL(((uint8_t *)mem1)[dst-1], 0);
            int cnt = dst;
            for(int k = 0; k < len; k++) {
                TEST_ASSERT_EQUAL(((uint8_t *)mem1)[dst+k], ((uint8_t*)(from[dst&3]))[cnt]);
                cnt = (cnt + 1) & 31;
            }
            TEST_ASSERT_EQUAL(((uint8_t *)mem1)[dst+len], 0);
        }
    }
//    printf("%d ticks\n", time);
    return errors;
}

void test_vpu_memset_256() {
  UNITY_SET_FILE();
  RUN_TEST(test_vpu_memset256);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_vpu_memset256();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
