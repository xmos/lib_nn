// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "nn_layers.h"
#include "nn_op_helper.h"

int requantize_int16_tensor_blob(void *blob, float input_scaler,
                                 float output_scaler, char *err_msg) {
    int16_t *output_tensor = (int16_t *)blob;
    const int tensor_length = 16;
    float combined_scaler = input_scaler / output_scaler;
    assert(combined_scaler > 0);
    int mult = round((combined_scaler - 1) * 32768);
    if (mult > 32767 || mult < -32768) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                 "Requantize FAIL! Input scaler is %g and output scaler is %g",
                 input_scaler, output_scaler);
        return 0;
    }
    for (int i = 0; i < tensor_length; i++) {
        output_tensor[i] = mult;
    }
    return 1;
}

#if NN_USE_REF
void requantize_int16_tensor_ref(int16_t *output, int16_t *input,
                                 int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *)blob;
    for (int i = 0; i < tensor_length; i++) {
        int64_t mult = (((int)input[i]) << 16) +
                       input[i] * (int64_t)multipliers[i & 15] * 2;
        mult = (mult + (1 << 15)) >> 16;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}
#else
extern void requantize_int16_tensor_asm(int16_t *output, int16_t *input,
                                        int tensor_length, void *blob);
#endif

void requantize_int16_tensor(int16_t *output, int16_t *input,
                             int tensor_length, void *blob) {
#ifdef NN_USE_REF
    requantize_int16_tensor_ref(output, input, tensor_length, blob);
#else
    requantize_int16_tensor_asm(output, input, tensor_length, blob);
#endif
}

#if CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

void requantize_16_to_8_ref(int8_t *y, const int16_t *x,
                                                        const unsigned elm_start,
                                                        const unsigned elm_count) {
    for (unsigned i = elm_start; i < elm_start + elm_count; i++) {
        y[i] = (x[i] < -0x7F80) ? NEG_SAT_VAL : vdepth8_single_s16(x[i]);
    }
}

#undef NEG_SAT_VAL

#ifdef NN_USE_REF
void requantize_16_to_8(int8_t *y, const int16_t *x,
                                                const unsigned elm_start, const unsigned elm_count) {
    requantize_16_to_8_ref(y, x, elm_start, elm_count);
}
#endif