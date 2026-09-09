// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "nn_layers.h"

int multiply_int16_tensor_blob(void *output, float input1_scaler,
                               float input2_scaler, float output_scaler,
                               char *err_msg) {
    int16_t *output_tensor = (int16_t *)output;
    float combined_scaler = input1_scaler * input2_scaler / output_scaler;
    assert(combined_scaler > 0);
    int shift = floor(log2(32768 / combined_scaler));
    int mult = ldexp(combined_scaler, shift);
    if (mult > 32767) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                 "Mul FAIL! Input1 scaler is %g, input2 scaler is %g, and output scaler is %g",
                 input1_scaler, input2_scaler, output_scaler);
        return 0;
    }
    output_tensor[0] = mult;
    output_tensor[1] = shift;
    return 1;
}

#if NN_USE_REF
void multiply_int16_tensor_ref(int16_t *output, int16_t *input1,
                               int16_t *input2, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *)blob;
    int shift = multipliers[1];
    for (int i = 0; i < tensor_length; i++) {
        int64_t mult = input1[i] * (int64_t)input2[i] * multipliers[0];
        mult = (mult + (1 << (shift - 1))) >> shift;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}
#else
extern void multiply_int16_tensor_asm(int16_t *output, int16_t *input1,
                                      int16_t *input2, int tensor_length,
                                      void *blob);
#endif

void multiply_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2,
                           int tensor_length, void *blob) {
#ifdef NN_USE_REF
    multiply_int16_tensor_ref(output, input1, input2, tensor_length, blob);
#else
    multiply_int16_tensor_asm(output, input1, input2, tensor_length, blob);
#endif
}

const int16_t eight_thousand[16] = {
    0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000,
    0x8000, 0x8000, 0x8000, 0x8000,
};