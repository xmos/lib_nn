#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "vpu_memmove_word_aligned.h"

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

int test_vpu_memmove_word_aligned(void) {
    int mem1[32];
    int mem2[32];
    int dst_vals[] = {0, 4, 28};
    int len_vals[] = {0, 1, 2, 3, 4, 5, 16, 30, 31, 32, 33, 34, 35, 36, 37, 38, 59, 60, 61, 62, 63};
    int errors = 0;
    for(int len_index = 0; len_index < sizeof(len_vals)/sizeof(int); len_index++) {
        int len = len_vals[len_index];
        for(int src = 0; src < 32; src+=28) {
            for(int dst_index = 0; dst_index < sizeof(dst_vals)/sizeof(int); dst_index++) {
                int dst = dst_vals[dst_index];
//                printf("Dst %d src %d len %d  2->1\n", dst, src, len);
                // First try mem2->mem1
                for(int k = 0; k < 32; k++) {
                    mem1[k] = 0;
                }
                for(int k = 0; k < len; k++) {
                    ((uint8_t *)mem2)[k+src] = k+0x40;
                }
                vpu_memmove_word_aligned(((uint8_t*)mem1)+dst, ((uint8_t*)mem2)+src, len);
                if (dst != 0) {
                    TEST_ASSERT_EQUAL(((uint8_t *)mem1)[dst-1], 0);
                }
                for(int k = 0; k < len; k++) {
                    TEST_ASSERT_EQUAL(((uint8_t *)mem1)[dst+k], k+0x40);
                }
                TEST_ASSERT_EQUAL(((uint8_t *)mem1)[dst+len], 0);

                // Now try mem1->mem2
//                printf("Dst %d src %d len %d  1->2\n", dst, src, len);

                for(int k = 0; k < 32; k++) {
                    mem2[k] = 0;
                }
                for(int k = 0; k < len; k++) {
                    ((uint8_t *)mem1)[k+src] = k+0x40;
                }
                vpu_memmove_word_aligned(((uint8_t*)mem2)+dst, ((uint8_t*)mem1)+src, len);
                if (dst != 0) {
                    TEST_ASSERT_EQUAL(((uint8_t *)mem2)[dst-1], 0);
                }
                for(int k = 0; k < len; k++) {
                    TEST_ASSERT_EQUAL(((uint8_t *)mem2)[dst+k], k+0x40);
                }
                TEST_ASSERT_EQUAL(((uint8_t *)mem2)[dst+len], 0);
                
//                printf("Dst %d src %d len %d  2->2\n", dst, src, len);
                // Now overlap mem2+src->mem2+dst
                for(int k = 0; k < 32; k++) {
                    mem2[k] = 0;
                }
                for(int k = 0; k < len; k++) {
                    ((uint8_t *)mem2)[k+src] = k+0x40;
                }
                vpu_memmove_word_aligned(((uint8_t*)mem2)+dst, ((uint8_t*)mem2)+src, len);
                if (dst != 0) {
                    TEST_ASSERT_EQUAL(((uint8_t *)mem2)[dst-1], src >= dst || len == 0 || dst-1-src >= len? 0x00: dst-1-src+0x40);
                }
                for(int k = 0; k < len; k++) {
                    TEST_ASSERT_EQUAL(((uint8_t *)mem2)[dst+k], k+0x40);
                }
                TEST_ASSERT_EQUAL(((uint8_t *)mem2)[dst+len], src <= dst || len == 0 || dst+len-src >= len || dst+len-src < 0 ? 0x00 : dst+len-src+0x40);
            }
        }
    }
    return errors;
}

void test_vpu_memmove_aligned() {
  UNITY_SET_FILE();
  RUN_TEST(test_vpu_memmove_word_aligned);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_vpu_memmove_word_aligned();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
