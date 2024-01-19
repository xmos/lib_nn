#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "multiply_int16.h"


#if 0

DEPRECATED
// Element multiplication between a tensor and a constant

extern void multiply_int16_constant_asm(int16_t *output, int16_t *input1, void *blob, int tensor_length);

void multiply_int16_constant_ref(int16_t *output, int16_t *input1, void *blob, int tensor_length) {
    int32_t *i_step_ptr = (int32_t *)blob;
    int i_step = i_step_ptr[0];
    int16_t *transformed_input2 = (int16_t *) &i_step_ptr[1];
    int higher_i = 0;
    for(int i = 0; i < tensor_length; i++) {
        int lower_i = i & 0xf;
        int transformed_index = higher_i << 1 | lower_i;
        int shift_index = higher_i << 1 | lower_i | 16;
        int mult = input1[i] * transformed_input2[transformed_index];
        mult = (mult + (1<<(transformed_input2[shift_index]-1))) >> transformed_input2[shift_index];
        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
        if ((i & 15) == 15) {
            higher_i += (32 + i_step) / 4;
        }
    }
}

void multiply_int16_constant(int16_t *output, int16_t *input1, void *transformed_input2, int tensor_length) {
#ifdef NN_USE_REF
    multiply_int16_constant_ref(output, input1, transformed_input2, tensor_length);
#else
    multiply_int16_constant_asm(output, input1, transformed_input2, tensor_length);
#endif
}

#endif


// Element multiplication between two tensors

extern void multiply_int16_tensor_asm(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob);

void multiply_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *) blob;
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult = input1[i] * (int64_t) input2[i] * multipliers[i & 15];
        mult = mult >> 29;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}

void multiply_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    multiply_int16_tensor_ref(output, input1, input2, tensor_length, blob);
#else
    multiply_int16_tensor_asm(output, input1, input2, tensor_length, blob);
#endif
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

extern void quantise_int16_tensor_asm(int16_t *output, int16_t *input1, void *blob, int tensor_length);

void quantise_int16_tensor_ref(int16_t *output, int16_t *input1, void *blob, int tensor_length) {
    int16_t *multipliers = (int16_t *) blob;
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult = (((int)input1[i]) << 16) + input1[i] * (int) multipliers[i & 15] * 2;
        mult = mult >> 16;

        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}

void quantise_int16_tensor(int16_t *output, int16_t *input1, void *blob, int tensor_length) {
#ifdef NN_USE_REF
    quantise_int16_tensor_ref(output, input1, blob, tensor_length);
#else
    quantise_int16_tensor_asm(output, input1, blob, tensor_length);
#endif
}


