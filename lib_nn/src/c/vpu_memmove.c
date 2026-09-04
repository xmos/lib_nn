// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <string.h>

#include "vpu_memmove.h"

#ifndef NN_USE_REF
extern void vpu_memmove_word_aligned_asm(void * dst, const void * src, unsigned byte_count);
#endif

void vpu_memmove_word_aligned(void * dst, const void * src, unsigned byte_count) {
#ifdef NN_USE_REF
    memmove(dst, src, byte_count);
#else
    vpu_memmove_word_aligned_asm(dst, src, byte_count);
#endif
}
