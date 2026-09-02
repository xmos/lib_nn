// Copyright 2020-2026 XMOS LIMITED.
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
#include "unity_fixture.h"
#include "vpu_memmove_word_aligned.h"
#include "vpu_memset_256.h"
#include "xs3_vpu.h"

TEST_GROUP(group_vpu);
TEST_SETUP(group_vpu) {}
TEST_TEAR_DOWN(group_vpu) {}
TEST_GROUP_RUNNER(group_vpu) {
  RUN_TEST_CASE(group_vpu, test_vpu_memcpy);
  RUN_TEST_CASE(group_vpu, test_vpu_memset_32);
  RUN_TEST_CASE(group_vpu, test_vpu_memset_vector);
  RUN_TEST_CASE(group_vpu, test_vpu_memmove_word_aligned);
  RUN_TEST_CASE(group_vpu, test_vpu_memset_256);
#ifdef TEST_BUILD_NATIVE
  RUN_TEST_CASE(group_vpu, test_vpu_memcpy_full);
#endif // TEST_BUILD_NATIVE
}

// ---------------------------------------------------------------------------
// vpu_memcpy / vpu_memcpy_int / vpu_memcpy_ext / vpu_memcpy_vector_{int,ext}
// ---------------------------------------------------------------------------

static void impl_vpu_memcpy_directed(size_t atom_bytes, int atom_count,
                                     int alignment, void (*mem_cpy_func)()) {
  size_t byte_count = atom_bytes * atom_count;

  int8_t* src_unaligned = (int8_t*)malloc(byte_count + alignment);
  int8_t* dst_unaligned = (int8_t*)malloc(byte_count + alignment);

  int8_t* src =
      src_unaligned + alignment - ((int64_t)src_unaligned % alignment);
  int8_t* dst =
      dst_unaligned + alignment - ((int64_t)dst_unaligned % alignment);

  int seed = 69;

  for (size_t b = 0; b < byte_count; b++) src[b] = (int8_t)pseudo_rand(&seed);

  memset(dst, 0, byte_count);
  mem_cpy_func(dst, src, atom_count);

  TEST_ASSERT_EQUAL_INT8_ARRAY(dst, src, byte_count);

  free(src_unaligned);
  free(dst_unaligned);
}

// A trimmed-down sweep of pointer/length combinations (the original swept
// every byte offset, which is far too slow to run under simulation).
static void impl_vpu_memcpy_pseudo_random(size_t src_pointer_inc,
                                          size_t dst_relative_pointer_inc,
                                          size_t atom_bytes, int atom_count,
                                          int alignment,
                                          void (*mem_cpy_func)()) {
  size_t byte_count = atom_bytes * atom_count;

  int8_t* src_unaligned = (int8_t*)malloc(byte_count + alignment);
  int8_t* dst_unaligned = (int8_t*)malloc(byte_count + alignment);

  int8_t* src =
      src_unaligned + alignment - ((int64_t)src_unaligned % alignment);
  int8_t* dst =
      dst_unaligned + alignment - ((int64_t)dst_unaligned % alignment);

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

TEST(group_vpu, test_vpu_memcpy) {
  impl_vpu_memcpy_directed(1, 256, 4, vpu_memcpy);
  impl_vpu_memcpy_directed(1, 256, 4, vpu_memcpy_int);
  impl_vpu_memcpy_directed(1, 256, 4, vpu_memcpy_ext);
  impl_vpu_memcpy_directed(MEMCPY_VECT_EXT_BYTES, 5, 4, vpu_memcpy_vector_ext);
  impl_vpu_memcpy_directed(MEMCPY_VECT_INT_BYTES, 5, 4, vpu_memcpy_vector_int);

  // Minimal smoke check of the pointer-offset logic (single offset, a
  // handful of lengths); the full sweep is far too slow under simulation,
  // so it's native-only.
  impl_vpu_memcpy_pseudo_random(32, 32, 1, 32, 4, vpu_memcpy);
  impl_vpu_memcpy_pseudo_random(32, 32, 1, 32, 4, vpu_memcpy_int);
  impl_vpu_memcpy_pseudo_random(32, 32, 1, 32, 4, vpu_memcpy_ext);
  impl_vpu_memcpy_pseudo_random(2, 2, MEMCPY_VECT_EXT_BYTES, 2, 4,
                                vpu_memcpy_vector_ext);
  impl_vpu_memcpy_pseudo_random(2, 2, MEMCPY_VECT_INT_BYTES, 2, 4,
                                vpu_memcpy_vector_int);
}

#ifdef TEST_BUILD_NATIVE
TEST(group_vpu, test_vpu_memcpy_full) {
  impl_vpu_memcpy_pseudo_random(8, 8, 1, 32, 4, vpu_memcpy);
  impl_vpu_memcpy_pseudo_random(8, 8, 1, 32, 4, vpu_memcpy_int);
  impl_vpu_memcpy_pseudo_random(8, 8, 1, 32, 4, vpu_memcpy_ext);
  impl_vpu_memcpy_pseudo_random(4, 4, MEMCPY_VECT_EXT_BYTES, 8, 4,
                                vpu_memcpy_vector_ext);
  impl_vpu_memcpy_pseudo_random(4, 4, MEMCPY_VECT_INT_BYTES, 8, 4,
                                vpu_memcpy_vector_int);
}
#endif // TEST_BUILD_NATIVE

// ---------------------------------------------------------------------------
// vpu_memset_32 / vpu_memset_vector
// ---------------------------------------------------------------------------

// A trimmed-down sweep (the original swept every word offset, which is far
// too slow to run under simulation).
static void impl_vpu_memset_32_pseudo_random(int pointer_inc,
                                             int set_words_inc,
                                             int max_test_vpu_words) {
  const size_t bytes_per_vpu_word = XS3_VPU_VREG_WIDTH_BYTES;
  const size_t mem_bytes = bytes_per_vpu_word * max_test_vpu_words;
  const int mem_words = mem_bytes / sizeof(int32_t);

  int32_t* dst = (int32_t*)malloc(bytes_per_vpu_word * max_test_vpu_words + 1);
  dst = (int32_t*)((char*)dst + (4 - (int64_t)dst & 3));

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

TEST(group_vpu, test_vpu_memset_32) {
#define DIR_TEST_0_WORDS 1024
  int64_t dst[DIR_TEST_0_WORDS / 2];
  int seed = 69;
  int32_t value = (int32_t)pseudo_rand(&seed);
  memset(dst, 0, DIR_TEST_0_WORDS * sizeof(int32_t));

  vpu_memset_32(dst, value, DIR_TEST_0_WORDS);

  TEST_ASSERT_EACH_EQUAL_INT32(value, dst, DIR_TEST_0_WORDS);
#undef DIR_TEST_0_WORDS

  impl_vpu_memset_32_pseudo_random(4, 4, 4);
}

// A trimmed-down sweep (the original swept every vector offset, which is far
// too slow to run under simulation).
static void impl_vpu_memset_vector_pseudo_random(int pointer_inc,
                                                 int set_vectors_inc,
                                                 int max_test_vpu_vectors) {
  const size_t bytes_per_vpu_vector = VPU_MEMSET_VECTOR_WORDS * sizeof(int);
  const size_t mem_bytes = bytes_per_vpu_vector * max_test_vpu_vectors;
  const int mem_words = mem_bytes / sizeof(int32_t);

  int32_t* dst =
      (int32_t*)malloc(bytes_per_vpu_vector * max_test_vpu_vectors + 1);
  dst = (int32_t*)((char*)dst + (4 - (int64_t)dst & 3));

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

TEST(group_vpu, test_vpu_memset_vector) {
#define DIR_TEST_0_VECTORS (8)
  int64_t dst[(VPU_MEMSET_VECTOR_WORDS * DIR_TEST_0_VECTORS) / 2];
  int seed = 69;
  int32_t value = (int32_t)pseudo_rand(&seed);
  memset(dst, 0,
         (VPU_MEMSET_VECTOR_WORDS * DIR_TEST_0_VECTORS) * sizeof(int32_t));

  vpu_memset_vector(dst, value, DIR_TEST_0_VECTORS);

  TEST_ASSERT_EACH_EQUAL_INT32(value, dst,
                               (VPU_MEMSET_VECTOR_WORDS * DIR_TEST_0_VECTORS));
#undef DIR_TEST_0_VECTORS

  impl_vpu_memset_vector_pseudo_random(2, 2, 4);
}

// ---------------------------------------------------------------------------
// vpu_memmove_word_aligned
// ---------------------------------------------------------------------------

static void impl_test_vpu_memmove_word_aligned(const int *len_vals,
                                               int len_count) {
  int mem1[32];
  int mem2[32];
  int dst_vals[] = {0, 4, 28};
  for (int len_index = 0; len_index < len_count; len_index++) {
    int len = len_vals[len_index];
    for (int src = 0; src < 32; src += 28) {
      for (int dst_index = 0; dst_index < sizeof(dst_vals) / sizeof(int);
           dst_index++) {
        int dst = dst_vals[dst_index];

        // mem2 -> mem1
        for (int k = 0; k < 32; k++) mem1[k] = 0;
        for (int k = 0; k < len; k++) ((uint8_t*)mem2)[k + src] = k + 0x40;
        vpu_memmove_word_aligned(((uint8_t*)mem1) + dst,
                                 ((uint8_t*)mem2) + src, len);
        if (dst != 0) TEST_ASSERT_EQUAL(((uint8_t*)mem1)[dst - 1], 0);
        for (int k = 0; k < len; k++)
          TEST_ASSERT_EQUAL(((uint8_t*)mem1)[dst + k], k + 0x40);
        TEST_ASSERT_EQUAL(((uint8_t*)mem1)[dst + len], 0);

        // mem1 -> mem2
        for (int k = 0; k < 32; k++) mem2[k] = 0;
        for (int k = 0; k < len; k++) ((uint8_t*)mem1)[k + src] = k + 0x40;
        vpu_memmove_word_aligned(((uint8_t*)mem2) + dst,
                                 ((uint8_t*)mem1) + src, len);
        if (dst != 0) TEST_ASSERT_EQUAL(((uint8_t*)mem2)[dst - 1], 0);
        for (int k = 0; k < len; k++)
          TEST_ASSERT_EQUAL(((uint8_t*)mem2)[dst + k], k + 0x40);
        TEST_ASSERT_EQUAL(((uint8_t*)mem2)[dst + len], 0);

        // mem2 -> mem2 (overlapping)
        for (int k = 0; k < 32; k++) mem2[k] = 0;
        for (int k = 0; k < len; k++) ((uint8_t*)mem2)[k + src] = k + 0x40;
        vpu_memmove_word_aligned(((uint8_t*)mem2) + dst,
                                 ((uint8_t*)mem2) + src, len);
        if (dst != 0)
          TEST_ASSERT_EQUAL(((uint8_t*)mem2)[dst - 1],
                            src >= dst || len == 0 || dst - 1 - src >= len
                                ? 0x00
                                : dst - 1 - src + 0x40);
        for (int k = 0; k < len; k++)
          TEST_ASSERT_EQUAL(((uint8_t*)mem2)[dst + k], k + 0x40);
        TEST_ASSERT_EQUAL(
            ((uint8_t*)mem2)[dst + len],
            src <= dst || len == 0 || dst + len - src >= len ||
                    dst + len - src < 0
                ? 0x00
                : dst + len - src + 0x40);
      }
    }
  }
}

TEST(group_vpu, test_vpu_memmove_word_aligned) {
  static const int len_vals[] = {0,  1,  2,  3,  4,  5,  16, 30, 31, 32, 33,
                                34, 35, 36, 37, 38, 59, 60, 61, 62, 63};
  impl_test_vpu_memmove_word_aligned(len_vals,
                                     sizeof(len_vals) / sizeof(int));
}

// ---------------------------------------------------------------------------
// vpu_memset_256
// ---------------------------------------------------------------------------

static void impl_test_vpu_memset_256(const int *len_vals, int len_count) {
  int mem1[32];
  uint64_t from[4][4];
  int from_ref[4][8] = {
      {0x80706050, 0x80706050, 0x80706050, 0x80706050, 0x80706050, 0x80706050,
       0x80706050, 0x80706050},  // int32
      {0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,
       0x80808080, 0x80808080},  // int8
      {0x80708070, 0x80708070, 0x80708070, 0x80708070, 0x80708070, 0x80708070,
       0x80708070, 0x80708070},  // int16
      {0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080, 0x80808080,
       0x80808080, 0x80808080},  // int8
  };
  broadcast_32_to_256(from[0], ((uint32_t*)from_ref[0])[0]);
  broadcast_32_to_256(from[1], BROADCAST_8_TO_32(((uint32_t*)from_ref[1])[0]));
  broadcast_32_to_256(from[2], BROADCAST_16_TO_32(((uint32_t*)from_ref[2])[0]));
  broadcast_32_to_256(from[3], BROADCAST_8_TO_32(((uint32_t*)from_ref[3])[0]));

  int dst_vals[] = {1, 2, 3, 4, 5};
  for (int len_index = 0; len_index < len_count; len_index++) {
    int len = len_vals[len_index];
    for (int dst_index = 0; dst_index < sizeof(dst_vals) / sizeof(int);
         dst_index++) {
      int dst = dst_vals[dst_index];

      for (int k = 0; k < 32; k++) mem1[k] = 0;

      vpu_memset_256(((uint8_t*)mem1) + dst, ((uint8_t*)(from[dst & 3])), len);

      TEST_ASSERT_EQUAL(((uint8_t*)mem1)[dst - 1], 0);
      int cnt = dst;
      for (int k = 0; k < len; k++) {
        TEST_ASSERT_EQUAL(((uint8_t*)mem1)[dst + k],
                          ((uint8_t*)(from_ref[dst & 3]))[cnt]);
        cnt = (cnt + 1) & 31;
      }
      TEST_ASSERT_EQUAL(((uint8_t*)mem1)[dst + len], 0);
    }
  }
}

TEST(group_vpu, test_vpu_memset_256) {
  static const int len_vals[] = {0,  1,  2,  3,  4,  5,  16, 30, 31, 32, 33,
                                34, 35, 36, 37, 38, 59, 60, 61, 62, 63};
  impl_test_vpu_memset_256(len_vals, sizeof(len_vals) / sizeof(int));
}
