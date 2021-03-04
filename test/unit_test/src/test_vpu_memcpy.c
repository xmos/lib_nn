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


void impl_vpu_memcpy_pseudo_random(
    size_t pointer_inc, 
    size_t cpy_bytes_inc,
    int max_test_vpu_words ){

    const size_t bytes_per_vpu_word = XS3_VPU_VREG_WIDTH_BYTES;

    const size_t mem_bytes = bytes_per_vpu_word * max_test_vpu_words;

    int8_t * src = malloc(bytes_per_vpu_word * max_test_vpu_words + 4);
    int8_t * dst = malloc(bytes_per_vpu_word * max_test_vpu_words + 4);
    
    //ensure that the src and dst pointers are word aligned
    src = (src + (4-(int)src&3));
    dst = (dst + (4-(int)dst&3));

    int seed = 69;
    for(size_t src_offset = 0; src_offset < mem_bytes; src_offset += pointer_inc){

        for(size_t dst_offset = src_offset; dst_offset < mem_bytes; dst_offset += sizeof(int)){

            size_t max_cpy_bytes = mem_bytes - dst_offset;

            for(size_t cpy_bytes = 1; cpy_bytes < max_cpy_bytes; cpy_bytes += cpy_bytes_inc){

                int8_t dst_init = (int8_t)pseudo_rand(&seed);

                memset(dst, dst_init, mem_bytes);

                memset(src, 0xff, mem_bytes);

                for(size_t b=0; b<cpy_bytes;b++)
                    src[b] = (int8_t)pseudo_rand(&seed);

                vpu_memcpy(dst + dst_offset, src + src_offset, cpy_bytes);

                TEST_ASSERT_EQUAL_INT8_ARRAY(dst + dst_offset, src + src_offset, cpy_bytes);
                if(dst_offset)
                    TEST_ASSERT_EACH_EQUAL_INT8(dst_init, dst, dst_offset);
                if (mem_bytes - dst_offset - cpy_bytes)
                    TEST_ASSERT_EACH_EQUAL_INT8(dst_init, dst + dst_offset + cpy_bytes, mem_bytes - dst_offset - cpy_bytes);
            }
        }
    }
}

void test_vpu_memcpy_vector_directed_0(){
    #define DIR_TEST_0_VECTOR_WORDS 5
    int8_t src[VPU_MEMCPU_VECTOR_BYTES*DIR_TEST_0_VECTOR_WORDS];
    int8_t dst[VPU_MEMCPU_VECTOR_BYTES*DIR_TEST_0_VECTOR_WORDS];   
    int seed = 69;             
    for(size_t b=0; b<VPU_MEMCPU_VECTOR_BYTES*DIR_TEST_0_VECTOR_WORDS;b++)
        src[b] = (int8_t)pseudo_rand(&seed);
    memset(dst, 0, sizeof(dst));

    vpu_memcpy_vector(dst, src, DIR_TEST_0_VECTOR_WORDS);

    TEST_ASSERT_EQUAL_INT8_ARRAY(dst, src, VPU_MEMCPU_VECTOR_BYTES*DIR_TEST_0_VECTOR_WORDS);
}

void impl_vpu_memcpy_vector_pseudo_random(
    size_t pointer_inc, 
    int max_test_vpu_words ){

    const size_t bytes_per_vpu_word = VPU_MEMCPU_VECTOR_BYTES;

    const size_t mem_bytes = bytes_per_vpu_word * max_test_vpu_words;

    int8_t * src = malloc(bytes_per_vpu_word * max_test_vpu_words + 4);
    int8_t * dst = malloc(bytes_per_vpu_word * max_test_vpu_words + 4);
    
    //ensure that the src and dst pointers are word aligned
    src = (src + (4-(int)src&3));
    dst = (dst + (4-(int)dst&3));

    int seed = 69;
    for(size_t src_offset = 0; src_offset < mem_bytes - VPU_MEMCPU_VECTOR_BYTES; src_offset += pointer_inc){

        for(size_t dst_offset = src_offset; dst_offset < mem_bytes - VPU_MEMCPU_VECTOR_BYTES; dst_offset += sizeof(int)){

            size_t max_cpy_vector_words = (mem_bytes - dst_offset) / VPU_MEMCPU_VECTOR_BYTES;

            for(size_t cpy_vector_words = 1; cpy_vector_words < max_cpy_vector_words; 
                cpy_vector_words += 1){

                int8_t dst_init = (int8_t)pseudo_rand(&seed);

                memset(dst, dst_init, mem_bytes);

                memset(src, 0xff, mem_bytes);

                size_t cpy_bytes = cpy_vector_words * VPU_MEMCPU_VECTOR_BYTES;

                for(size_t b=0; b<cpy_bytes;b++)
                    src[b] = (int8_t)pseudo_rand(&seed);

                vpu_memcpy_vector(dst + dst_offset, src + src_offset, cpy_vector_words);

                TEST_ASSERT_EQUAL_INT8_ARRAY(dst + dst_offset, src + src_offset, cpy_bytes);
                if(dst_offset)
                    TEST_ASSERT_EACH_EQUAL_INT8(dst_init, dst, dst_offset);
                if (mem_bytes - dst_offset - cpy_bytes)
                    TEST_ASSERT_EACH_EQUAL_INT8(dst_init, dst + dst_offset + cpy_bytes, mem_bytes - dst_offset - cpy_bytes);
            }
        }
    }
}

void test_vpu_memcpy_pseudo_random(){
    impl_vpu_memcpy_pseudo_random(1, 1, 5);
}

void test_vpu_memcpy_vector_pseudo_random(){
    impl_vpu_memcpy_vector_pseudo_random(4, 7);
}

void test_vpu_memcpy()
{

    UNITY_SET_FILE();

    RUN_TEST(test_vpu_memcpy_directed_0);
    RUN_TEST(test_vpu_memcpy_pseudo_random);
    RUN_TEST(test_vpu_memcpy_vector_directed_0);
    RUN_TEST(test_vpu_memcpy_vector_pseudo_random);
}