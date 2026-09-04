// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "nn_layers.h"



// Element multiplication between two tensors

#ifdef NN_USE_REF
void multiply_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *) blob;
    int shift = multipliers[1];
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult = input1[i] * (int64_t) input2[i] * multipliers[0];
        mult = (mult + (1 << (shift-1))) >> shift;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}

#else

extern void multiply_int16_tensor_asm(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob);

#endif

void multiply_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    multiply_int16_tensor_ref(output, input1, input2, tensor_length, blob);
#else
    multiply_int16_tensor_asm(output, input1, input2, tensor_length, blob);
#endif
}
