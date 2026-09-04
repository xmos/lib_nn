// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "nn_layers.h"

// Element quantization of float to int16_t tensor

#ifdef NN_USE_REF
void quantize_int16_tensor_ref(int16_t *output, float *input, int tensor_length, void *blob) {
    for(int i = 0; i < tensor_length; i++) {
        float a = input[i] * ((float *) blob)[0];
        a = ldexp(a, 23);
        a = floor(a+0.5);           // Do not use ROUND as that rounds tie to even
        if (a >  32767) a =  32767;
        if (a < -32768) a = -32768;
        output[i] = a;
    }
}

#else

extern void quantize_int16_tensor_asm(int16_t *output, float *input, int tensor_length, void *blob);

#endif

void quantize_int16_tensor(int16_t *output, float *input1, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    quantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
    quantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}
