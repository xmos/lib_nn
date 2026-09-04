// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "nn_layers.h"

#define SHIFT 14

// Element multiplication between two tensors

#ifdef NN_USE_REF
void add_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *) blob;
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult1 = input1[i] * (int64_t) multipliers[(i & 15)     ];
        int64_t mult2 = input2[i] * (int64_t) multipliers[(i & 15) + 16];
        int answer = (mult1 + mult2 + (1<<(SHIFT-1))) >> SHIFT;

        if (answer > 32767) answer = 32767;
        if (answer < -32768) answer = -32768;
        output[i] = answer;
    }
}

#else

extern void add_int16_tensor_asm(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob);

#endif

void add_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    add_int16_tensor_ref(output, input1, input2, tensor_length, blob);
#else
    add_int16_tensor_asm(output, input1, input2, tensor_length, blob);
#endif
}
