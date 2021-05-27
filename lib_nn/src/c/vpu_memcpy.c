// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <string.h>

#include "nn_op_utils.h"

#ifdef NN_USE_REF

void vpu_memcpy(void* dst, const void* src, size_t byte_count) {
  memcpy(dst, src, byte_count);
}

void vpu_memcpy_int(void* dst, const void* src, size_t byte_count) {
  memcpy(dst, src, byte_count);
}

void vpu_memcpy_ext(void* dst, const void* src, size_t byte_count) {
  memcpy(dst, src, byte_count);
}

void vpu_memcpy_vector_ext(void* dst, const void* src, int vector_count) {
  memcpy(dst, src, vector_count * MEMCPY_VECT_EXT_BYTES);
}

void vpu_memcpy_vector_int(void* dst, const void* src, int vector_count) {
  memcpy(dst, src, vector_count * MEMCPY_VECT_INT_BYTES);
}

#else

static inline void vpu_memcpy_base(void* dst, const void* src,
                                   size_t byte_count, void (*mem_cpy_func)(),
                                   size_t vector_bytes) {
  // The code below doesnt support such small copies
  if (byte_count < 4) {
    memcpy(dst, src, byte_count);
    return;
  }

  // src and dst alignment must be the same
  assert(((int)dst & 0x3) == ((int)src & 0x3));

  // The head is from the src address to the first word aligned address
  int alignment = (int)src & 0x3;
  if (alignment) {
    size_t head_bytes = 4 - alignment;
    byte_count -= head_bytes;
    memcpy(dst, src, head_bytes);
    dst += head_bytes;
    src += head_bytes;
  }

  // body
  int vector_count = byte_count / vector_bytes;
  mem_cpy_func(dst, src, vector_count);
  size_t vpy_memcpy_bytes = vector_bytes * vector_count;
  dst += vpy_memcpy_bytes;
  src += vpy_memcpy_bytes;
  byte_count -= vpy_memcpy_bytes;

  // tail
  size_t tail_bytes = byte_count;
  memcpy(dst, src, tail_bytes);
}

void vpu_memcpy_vector_ext_asm(void* dst, const void* src, size_t byte_count);
void vpu_memcpy_vector_int_asm(void* dst, const void* src, size_t byte_count);

void vpu_memcpy(void* dst, const void* src, size_t byte_count) {
  vpu_memcpy_base(dst, src, byte_count, vpu_memcpy_vector_ext_asm,
                  MEMCPY_VECT_EXT_BYTES);
}

void vpu_memcpy_int(void* dst, const void* src, size_t byte_count) {
  vpu_memcpy_base(dst, src, byte_count, vpu_memcpy_vector_int_asm,
                  MEMCPY_VECT_INT_BYTES);
}

void vpu_memcpy_ext(void* dst, const void* src, size_t byte_count) {
  vpu_memcpy_base(dst, src, byte_count, vpu_memcpy_vector_ext_asm,
                  MEMCPY_VECT_EXT_BYTES);
}

void vpu_memcpy_vector_ext(void* dst, const void* src, int vector_count) {
  assert(((int)dst & 0x3) == 0);

  vpu_memcpy_vector_ext_asm(dst, src, vector_count);
}

void vpu_memcpy_vector_int(void* dst, const void* src, int vector_count) {
  assert(((int)dst & 0x3) == 0);

  vpu_memcpy_vector_int_asm(dst, src, vector_count);
}

#endif  // NN_USE_REF
