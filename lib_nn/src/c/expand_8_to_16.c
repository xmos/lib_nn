// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "expand_8_to_16.h"

extern void expand_8_to_16_ref_asm(int16_t *out, int8_t *in, int N);

void expand_8_to_16_ref(int16_t *out, int8_t *in, int N) {
    for(int i = 0; i < N; i++) {
        out[i] = in[i];
    }
}

void expand_8_to_16(int16_t *out, int8_t *in, int N) {
#ifdef NN_USE_REF
    expand_8_to_16_ref(out, in, N);
#else
    expand_8_to_16_ref_asm(out, in, N);
#endif
}
