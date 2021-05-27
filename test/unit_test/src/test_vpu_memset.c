// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"
#include "nn_operator.h"
#include "tst_common.h"
#include "unity.h"
#include "xs3_vpu.h"

void test_vpu_memset_32_directed_0() {
#define DIR_TEST_0_WORDS 1024
  int64_t dst[DIR_TEST_0_WORDS / 2];
  int seed = 69;
  int32_t value = (int32_t)pseudo_rand(&seed);
  memset(dst, 0, DIR_TEST_0_WORDS * sizeof(int32_t));

  vpu_memset_32(dst, value, DIR_TEST_0_WORDS);

  TEST_ASSERT_EACH_EQUAL_INT32(value, dst, DIR_TEST_0_WORDS);
}

void impl_vpu_memset_32_pseudo_random(int pointer_inc, int set_words_inc,
                                      int max_test_vpu_words) {
  const size_t bytes_per_vpu_word = XS3_VPU_VREG_WIDTH_BYTES;

  const size_t mem_bytes = bytes_per_vpu_word * max_test_vpu_words;

  const int mem_words = mem_bytes / sizeof(int32_t);

  int32_t *dst = (int32_t *)malloc(bytes_per_vpu_word * max_test_vpu_words + 1);

  // align the dst pointer
  dst = (int32_t *)((char *)dst + (4 - (int)dst & 3));

  int seed = 69;

  for (int dst_offset = 0; dst_offset < mem_words; dst_offset += pointer_inc) {
    int max_set_words = mem_words - dst_offset;

    for (int set_words = 4; set_words < max_set_words;
         set_words += set_words_inc) {
      int32_t value = (int32_t)pseudo_rand(&seed);
      int8_t init_value = (int8_t)pseudo_rand(&seed);

      for (unsigned i = 0; i < mem_words; i++) dst[i] = init_value;

      vpu_memset_32(dst + dst_offset, value, set_words);

      TEST_ASSERT_EACH_EQUAL_INT32(value, dst + dst_offset, set_words);

      if (dst_offset) TEST_ASSERT_EACH_EQUAL_INT32(init_value, dst, dst_offset);

      if (mem_words - dst_offset - set_words)
        TEST_ASSERT_EACH_EQUAL_INT32(init_value, dst + dst_offset + set_words,
                                     mem_words - dst_offset - set_words);
    }
  }
}

void test_vpu_memset_vector_directed_0() {
#define DIR_TEST_0_VECTORS (8)
  int64_t dst[(VPU_MEMSET_VECTOR_WORDS * DIR_TEST_0_VECTORS) / 2];
  int seed = 69;
  int32_t value = (int32_t)pseudo_rand(&seed);
  memset(dst, 0,
         (VPU_MEMSET_VECTOR_WORDS * DIR_TEST_0_VECTORS) * sizeof(int32_t));

  vpu_memset_vector(dst, value, DIR_TEST_0_VECTORS);

  TEST_ASSERT_EACH_EQUAL_INT32(value, dst,
                               (VPU_MEMSET_VECTOR_WORDS * DIR_TEST_0_VECTORS));
}

void impl_vpu_memset_vector_pseudo_random(int pointer_inc, int set_vectors_inc,
                                          int max_test_vpu_vectors) {
  const size_t bytes_per_vpu_vector = VPU_MEMSET_VECTOR_WORDS * sizeof(int);

  const size_t mem_bytes = bytes_per_vpu_vector * max_test_vpu_vectors;

  const int mem_words = mem_bytes / sizeof(int32_t);

  int32_t *dst =
      (int32_t *)malloc(bytes_per_vpu_vector * max_test_vpu_vectors + 1);

  // align the dst pointer
  dst = (int32_t *)((char *)dst + (4 - (int)dst & 3));

  int seed = 69;

  for (int dst_offset = 0; dst_offset < mem_words - VPU_MEMSET_VECTOR_WORDS;
       dst_offset += pointer_inc) {
    int max_set_vectors = (mem_words - dst_offset) / VPU_MEMSET_VECTOR_WORDS;

    for (int set_vectors = 1; set_vectors < max_set_vectors;
         set_vectors += set_vectors_inc) {
      int32_t value = (int32_t)pseudo_rand(&seed);
      int8_t init_value = (int8_t)pseudo_rand(&seed);

      for (unsigned i = 0; i < mem_words; i++) dst[i] = init_value;

      vpu_memset_vector(dst + dst_offset, value, set_vectors);

      int set_words = set_vectors * VPU_MEMSET_VECTOR_WORDS;
      TEST_ASSERT_EACH_EQUAL_INT32(value, dst + dst_offset, set_words);

      if (dst_offset) TEST_ASSERT_EACH_EQUAL_INT32(init_value, dst, dst_offset);

      if (mem_words - dst_offset - set_words)
        TEST_ASSERT_EACH_EQUAL_INT32(init_value, dst + dst_offset + set_words,
                                     mem_words - dst_offset - set_words);
    }
  }
}
void test_vpu_memset_32_pseudo_random() {
  impl_vpu_memset_32_pseudo_random(1, 1, 20);
}
void test_vpu_memset_vector_pseudo_random() {
  impl_vpu_memset_vector_pseudo_random(1, 1, 20);
}

void test_vpu_memset() {
  UNITY_SET_FILE();

  RUN_TEST(test_vpu_memset_32_directed_0);
  RUN_TEST(test_vpu_memset_32_pseudo_random);
  RUN_TEST(test_vpu_memset_vector_directed_0);
  RUN_TEST(test_vpu_memset_vector_pseudo_random);
}