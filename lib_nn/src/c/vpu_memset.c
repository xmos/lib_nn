// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "nn_op_utils.h"

#ifdef NN_USE_REF

void vpu_memset_vector(void *dst, const int32_t value, const int vector_count) {
  vpu_memset_32(dst, value, vector_count * VPU_MEMSET_VECTOR_WORDS);
}

void vpu_memset_32(void *dst, const int32_t value, const int word_count) {
  int32_t *dst32 = (int32_t *)dst;
  for (int i = 0; i < word_count; i++) dst32[i] = value;
}

#else

void vpu_memset32_asm(void *dst, const int32_t value, const int itts);

void vpu_memset_vector(void *dst, const int32_t value, const int vector_count) {
  assert(((int)dst & 0x3) == 0);

  int32_t *dst32 = (int32_t *)dst;
  vpu_memset32_asm(dst32, value, vector_count);
}

void vpu_memset_32(void *dst, const int32_t value, const int word_count) {
  assert(((int)dst & 0x3) == 0);

  int32_t *dst32 = (int32_t *)dst;

  // do the leading words
  unsigned leading_words = word_count % VPU_MEMSET_VECTOR_WORDS;
  for (int i = 0; i < leading_words; i++) dst32[i] = value;

  dst32 += leading_words;
  int remaining_words = word_count - leading_words;

  assert(remaining_words % VPU_MEMSET_VECTOR_WORDS == 0);

  int vector_count = remaining_words / VPU_MEMSET_VECTOR_WORDS;
  vpu_memset_vector(dst32, value, vector_count);
}
#endif  // NN_USE_REF