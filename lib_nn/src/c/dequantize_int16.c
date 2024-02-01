#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "dequantize_int16.h"

#define SHIFT 14

// Element multiplication between two tensors

extern void dequantize_int16_tensor_asm(float *output, int16_t *input, int tensor_length, void *blob);

void dequantize_int16_tensor_ref(float *output, int16_t *input, int tensor_length, void *blob) {
    for(int i = 0; i < tensor_length; i++) {
        float a;
        *(int *) & a = input[i] + 0x8000;
        output[i] = a * (double) ((float *) blob)[0] * 8192 +  (double) ((float *) blob)[1];
    }
}

void dequantize_int16_tensor(float *output, int16_t *input1, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    dequantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
    dequantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}

