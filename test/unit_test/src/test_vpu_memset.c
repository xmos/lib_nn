// Copyright 2020 XMOS LIMITED. This Software is subject to the terms of the 
// XMOS Public License: Version 1

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"
#include "helpers.h"

void test_vpu_memset_directed_0(){
    #define DIR_TEST_0_WORDS 1024
    int64_t dst[DIR_TEST_0_WORDS/2];   
    int seed = 69;             
    int32_t value = (int32_t)pseudo_rand(&seed);
    memset(dst, 0, DIR_TEST_0_WORDS*sizeof(int32_t));

    vpu_memset_32(dst, value, DIR_TEST_0_WORDS);

    TEST_ASSERT_EACH_EQUAL_INT32(value, dst, DIR_TEST_0_WORDS);
}

void impl_vpu_memset_pseudo_random(
    int pointer_inc, 
    int set_words_inc,
    int max_test_vpu_words ){

    const size_t bytes_per_vpu_word = XS3_VPU_VREG_WIDTH_BYTES;

    const size_t mem_bytes = bytes_per_vpu_word * max_test_vpu_words;

    const int mem_words = mem_bytes / sizeof(int32_t);

    int32_t * dst = (int32_t *)malloc(bytes_per_vpu_word * max_test_vpu_words + 1);
    
    dst = (char*)dst + (4-(int)dst&3);

    int seed = 69;

    for(int dst_offset = 0; dst_offset < mem_words; dst_offset += pointer_inc){

        int max_set_words = mem_words - dst_offset;

        for(int set_words = 4; set_words < max_set_words; set_words += set_words_inc){

            int32_t value = (int32_t)pseudo_rand(&seed);
            int8_t init_value = (int8_t)pseudo_rand(&seed);
            
            for (unsigned i=0;i<mem_words;i++)
                dst[i] = init_value;
            
            vpu_memset_32(dst + dst_offset, value, set_words);

            TEST_ASSERT_EACH_EQUAL_INT32(value, dst + dst_offset, set_words);
            
            if(dst_offset)
                TEST_ASSERT_EACH_EQUAL_INT32(init_value, dst, dst_offset);

            if (mem_words - dst_offset - set_words)
                TEST_ASSERT_EACH_EQUAL_INT32(init_value, dst + dst_offset + set_words, mem_words - dst_offset - set_words);
        }
    }
}

void test_vpu_memset_pseudo_random(){
    impl_vpu_memset_pseudo_random(1, 1, 20);
}

void test_vpu_memset()
{

    UNITY_SET_FILE();

    RUN_TEST(test_vpu_memset_directed_0);
    RUN_TEST(test_vpu_memset_pseudo_random);
}