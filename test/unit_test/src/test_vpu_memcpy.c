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


void test_vpu_memcpy_directed_0(){
    #define DIR_TEST_0_BYTES 1024
    int8_t src[DIR_TEST_0_BYTES];
    int8_t dst[DIR_TEST_0_BYTES];   
    int seed = 69;             
    for(size_t b=0; b<DIR_TEST_0_BYTES;b++)
        src[b] = (int8_t)pseudo_rand(&seed);
    memset(dst, 0, DIR_TEST_0_BYTES);

    vpu_memcpy(dst, src, DIR_TEST_0_BYTES);

    TEST_ASSERT_EQUAL_INT8_ARRAY(dst, src, DIR_TEST_0_BYTES);
}

void test_vpu_memcpy_pseudo_random(){

    const size_t bytes_per_vpu_word = 32;
    const size_t max_test_vpu_words = 24;

    const size_t mem_bytes = bytes_per_vpu_word * max_test_vpu_words;
    int8_t * src = malloc(bytes_per_vpu_word * max_test_vpu_words + 4);
    int8_t * dst = malloc(bytes_per_vpu_word * max_test_vpu_words + 4);
    
    //ensure that the src and dst pointers are word aligned
    if ((int)src&3) src += (3 - ((int)src&3));
    if ((int)dst&3) dst += (3 - ((int)dst&3));

    int seed = 69;
    for(size_t src_offset = 0; src_offset < mem_bytes; src_offset += 4){

        for(size_t dst_offset = src_offset; dst_offset < mem_bytes; dst_offset += 4){

            size_t max_cpy_bytes = mem_bytes - dst_offset;

            for(size_t cpy_bytes = 4; cpy_bytes < max_cpy_bytes; cpy_bytes += 4){
                memset(dst, 0x0, mem_bytes);

                memset(src, 0xff, mem_bytes);

                for(size_t b=0; b<cpy_bytes;b++)
                    src[b] = (int8_t)pseudo_rand(&seed);

                vpu_memcpy(dst + dst_offset, src + src_offset, cpy_bytes);

                TEST_ASSERT_EQUAL_INT8_ARRAY(dst + dst_offset, src + src_offset, cpy_bytes);
            }
        }
    }
}


void test_vpu_memcpy()
{

    UNITY_SET_FILE();

    RUN_TEST(test_vpu_memcpy_directed_0);
    RUN_TEST(test_vpu_memcpy_pseudo_random);
}