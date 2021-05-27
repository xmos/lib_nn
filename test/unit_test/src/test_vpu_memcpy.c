// Copyright 2020-2021 XMOS LIMITED.
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

void impl_vpu_memcpy_directed(size_t atom_bytes, int atom_count, int alignment,
                              void (*mem_cpy_func)()) {
  size_t byte_count = atom_bytes * atom_count;

  int8_t* src_unaligned = (int8_t*)malloc(byte_count + alignment);
  int8_t* dst_unaligned = (int8_t*)malloc(byte_count + alignment);

  int8_t* src = src_unaligned + alignment - ((int)src_unaligned % alignment);
  int8_t* dst = dst_unaligned + alignment - ((int)dst_unaligned % alignment);

  int seed = 69;

  for (size_t b = 0; b < byte_count; b++) src[b] = (int8_t)pseudo_rand(&seed);

  memset(dst, 0, byte_count);
  mem_cpy_func(dst, src, atom_count);

  TEST_ASSERT_EQUAL_INT8_ARRAY(dst, src, byte_count);

  free(src_unaligned);
  free(dst_unaligned);
}

void impl_vpu_memcpy_pseudo_random(size_t src_pointer_inc,
                                   size_t dst_relative_pointer_inc,

                                   size_t atom_bytes, int atom_count,
                                   int alignment, void (*mem_cpy_func)()) {
  size_t byte_count = atom_bytes * atom_count;

  int8_t* src_unaligned = (int8_t*)malloc(byte_count + alignment);
  int8_t* dst_unaligned = (int8_t*)malloc(byte_count + alignment);

  int8_t* src = src_unaligned + alignment - ((int)src_unaligned % alignment);
  int8_t* dst = dst_unaligned + alignment - ((int)dst_unaligned % alignment);

  int seed = 69;

  for (size_t src_offset = 0; src_offset < byte_count - atom_bytes;
       src_offset += src_pointer_inc) {
    for (size_t dst_offset = src_offset; dst_offset < byte_count - atom_bytes;
         dst_offset += dst_relative_pointer_inc) {
      size_t max_cpy_atoms = (byte_count - dst_offset) / atom_bytes;

      for (size_t cpy_atoms = 1; cpy_atoms < max_cpy_atoms; cpy_atoms += 1) {
        int8_t dst_init = (int8_t)pseudo_rand(&seed);

        memset(dst, dst_init, byte_count);

        memset(src, 0xff, byte_count);

        size_t cpy_bytes = cpy_atoms * atom_bytes;

        for (size_t b = 0; b < cpy_bytes; b++)
          src[b] = (int8_t)pseudo_rand(&seed);

        mem_cpy_func(dst + dst_offset, src + src_offset, cpy_atoms);

        TEST_ASSERT_EQUAL_INT8_ARRAY(dst + dst_offset, src + src_offset,
                                     cpy_bytes);
        if (dst_offset) TEST_ASSERT_EACH_EQUAL_INT8(dst_init, dst, dst_offset);
        if (byte_count - dst_offset - cpy_bytes)
          TEST_ASSERT_EACH_EQUAL_INT8(dst_init, dst + dst_offset + cpy_bytes,
                                      byte_count - dst_offset - cpy_bytes);
      }
    }
  }

  free(src_unaligned);
  free(dst_unaligned);
}

void test_vpu_memcpy_pseudo_random() {
  impl_vpu_memcpy_pseudo_random(1, 4, 1, 256, 4, vpu_memcpy);
}

void test_vpu_memcpy_pseudo_random_int() {
  impl_vpu_memcpy_pseudo_random(1, 4, 1, 256, 4, vpu_memcpy_int);
}

void test_vpu_memcpy_pseudo_random_ext() {
  impl_vpu_memcpy_pseudo_random(1, 4, 1, 256, 4, vpu_memcpy_ext);
}

void test_vpu_memcpy_pseudo_random_vector_ext() {
  impl_vpu_memcpy_pseudo_random(4, 4, MEMCPY_VECT_EXT_BYTES, 8, 4,
                                vpu_memcpy_vector_ext);
}

void test_vpu_memcpy_pseudo_random_vector_int() {
  impl_vpu_memcpy_pseudo_random(4, 4, MEMCPY_VECT_INT_BYTES, 8, 4,
                                vpu_memcpy_vector_int);
}

void test_vpu_memcpy_directed() {
  impl_vpu_memcpy_directed(1, 256, 4, vpu_memcpy);
}

void test_vpu_memcpy_directed_int() {
  impl_vpu_memcpy_directed(1, 256, 4, vpu_memcpy_int);
}

void test_vpu_memcpy_directed_ext() {
  impl_vpu_memcpy_directed(1, 256, 4, vpu_memcpy_ext);
}

void test_vpu_memcpy_directed_vector_ext() {
  impl_vpu_memcpy_directed(MEMCPY_VECT_EXT_BYTES, 5, 4, vpu_memcpy_vector_ext);
}

void test_vpu_memcpy_directed_vector_int() {
  impl_vpu_memcpy_directed(MEMCPY_VECT_INT_BYTES, 5, 4, vpu_memcpy_vector_int);
}

void test_vpu_memcpy() {
  UNITY_SET_FILE();
  RUN_TEST(test_vpu_memcpy_directed);
  RUN_TEST(test_vpu_memcpy_directed_int);
  RUN_TEST(test_vpu_memcpy_directed_ext);
  RUN_TEST(test_vpu_memcpy_directed_vector_int);
  RUN_TEST(test_vpu_memcpy_directed_vector_ext);
  RUN_TEST(test_vpu_memcpy_pseudo_random);
  RUN_TEST(test_vpu_memcpy_pseudo_random_int);
  RUN_TEST(test_vpu_memcpy_pseudo_random_ext);
  RUN_TEST(test_vpu_memcpy_pseudo_random_vector_ext);
  RUN_TEST(test_vpu_memcpy_pseudo_random_vector_int);
}