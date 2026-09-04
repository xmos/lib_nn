// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "vpu_memset.h"

#ifdef NN_USE_REF

void vpu_memset_vector(void *dst, const int32_t value, const int vector_count) {
  vpu_memset_32(dst, value, vector_count * VPU_MEMSET_VECTOR_WORDS);
}

void vpu_memset_32(void *dst, const int32_t value, const int word_count) {
  int32_t *dst32 = (int32_t *)dst;
  for (int i = 0; i < word_count; i++) dst32[i] = value;
}

void vpu_memset_256_ref(void *dst, const void *src, unsigned byte_count) {
  int s = (int)(((uintptr_t)dst) & 3);
  for (unsigned i = 0; i < byte_count; i++) {
    ((uint8_t *)dst)[i] = ((uint8_t *)src)[s];
    s = (s + 1) & 31;
  }
}

void vpu_memset_256(void *dst, const void *src, unsigned byte_count) {
  vpu_memset_256_ref(dst, src, byte_count);
}

void broadcast_32_to_256(void *dst, uint32_t from) {
  for (int i = 0; i < 8; i++) {
    ((uint32_t *)dst)[i] = from;
  }
}

#else

extern void vpu_memset32_asm(void *dst, const int32_t value, const int itts);
extern void vpu_memset_256_asm(void *dst, const void *src, unsigned byte_count);

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
  for (unsigned i = 0; i < leading_words; i++) dst32[i] = value;

  dst32 += leading_words;
  int remaining_words = word_count - leading_words;

  assert(remaining_words % VPU_MEMSET_VECTOR_WORDS == 0);

  int vector_count = remaining_words / VPU_MEMSET_VECTOR_WORDS;
  vpu_memset_vector(dst32, value, vector_count);
}

void vpu_memset_256(void *dst, const void *src, unsigned byte_count) {
  vpu_memset_256_asm(dst, src, byte_count);
}

void broadcast_32_to_256(void *dst, uint32_t from) {
#if defined(__XS3A__)
  asm("std %0, %1, %2[0]" :: "r" (from), "r" (from), "r" (dst));
  asm("std %0, %1, %2[1]" :: "r" (from), "r" (from), "r" (dst));
  asm("std %0, %1, %2[2]" :: "r" (from), "r" (from), "r" (dst));
  asm("std %0, %1, %2[3]" :: "r" (from), "r" (from), "r" (dst));
#endif
#if defined(__VX4A__) || defined(__VX4B__)
  for (int i = 0; i < 8; i++) {
    ((uint32_t *)dst)[i] = from;
  }
#endif
}
#endif  // NN_USE_REF
