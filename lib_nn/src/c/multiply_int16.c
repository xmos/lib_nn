#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "multiply_int16.h"



// Element multiplication between two tensors

extern void multiply_int16_tensor_asm(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob);

void multiply_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *) blob;
    int16_t shift = multipliers[16];
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult = input1[i] * (int64_t) input2[i] * multipliers[i & 15];
        mult = mult >> shift;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}

void multiply_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
//#ifdef NN_USE_REF
    multiply_int16_tensor_ref(output, input1, input2, tensor_length, blob);
// #else
//     multiply_int16_tensor_asm(output, input1, input2, tensor_length, blob);
// #endif
}


const int16_t eight_thousand[16] = {
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
    0x8000,
};



// Element multiplication between two tensors

extern void requantize_int16_tensor_asm(int16_t *output, int16_t *input1, int tensor_length, void *blob);

void requantize_int16_tensor_ref(int16_t *output, int16_t *input1, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *) blob;
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult = (((int)input1[i]) << 16) + input1[i] * (int) multipliers[i & 15] * 2;
        mult = mult >> 16;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}

void requantize_int16_tensor(int16_t *output, int16_t *input1, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    requantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
    requantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}


