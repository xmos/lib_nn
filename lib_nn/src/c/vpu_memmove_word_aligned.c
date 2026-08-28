// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <string.h>
#include "vpu_memmove_word_aligned.h"

#ifdef NN_USE_REF
void vpu_memmove_word_aligned(void * dst, const void * src, unsigned byte_count) {
    memmove(dst, src, byte_count);
}
#endif
