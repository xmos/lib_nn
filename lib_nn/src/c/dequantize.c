// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "nn_layers.h"

int dequantize_int16_tensor_blob(void *output,
                                 float input_scaler,
                                 char *err_msg) {
    float *blob = (float *)output;
    blob[0] = input_scaler * (1<<22);
    if (isinf(blob[0])) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                "Dequantize FAIL! Input scaler is %g",
                input_scaler);
        return 0;
    }
    ((int *)blob)[1] = 0xc0008000;         // This is -0x40008000 which is added during conversion
    return 1;
}

#if NN_USE_REF
void dequantize_int16_tensor_ref(float *output, int16_t *input, int tensor_length, void *blob) {
    for(int i = 0; i < tensor_length; i++) {
        float a;
        int bits = input[i] + 0x40008000;
        memcpy(&a, &bits, sizeof(a));
        output[i] = (a +  (double) ((float *) blob)[1]) * (double) ((float *) blob)[0] ;
    }
}
#else
extern 
void dequantize_int16_tensor_asm(float *output, int16_t *input, int tensor_length, void *blob);
#endif

void dequantize_int16_tensor(float *output, int16_t *input1, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    dequantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
    dequantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}
