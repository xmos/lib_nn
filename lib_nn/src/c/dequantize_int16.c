// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "dequantize_int16.h"

// Element dequantisaition
// Convert an int to a float without a cast, by adding 0x40008000
// that makes it a number in the range 2 + 1/64 + intvalue / 2^22
// Then subtract 0x40008000 away (as a float value), and multiply by 2^22 and a scalar
// These two values are precomputed in a blob.
// The clever bit is that we avoid normalisation and let the float addition take care of that.

#ifdef NN_USE_REF
void dequantize_int16_tensor_ref(float *output, int16_t *input, int tensor_length, void *blob) {
    for(int i = 0; i < tensor_length; i++) {
        float a;
        int bits = input[i] + 0x40008000;
        memcpy(&a, &bits, sizeof(a));
        output[i] = (a +  (double) ((float *) blob)[1]) * (double) ((float *) blob)[0] ;
    }
}

#else

extern void dequantize_int16_tensor_asm(float *output, int16_t *input, int tensor_length, void *blob);

#endif

void dequantize_int16_tensor(float *output, int16_t *input1, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    dequantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
    dequantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}
