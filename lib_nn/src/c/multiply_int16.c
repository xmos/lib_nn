#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "multiply_int16.h"

extern void multiply_int16_elementwise_constant_asm(int16_t *output, int16_t *input1, void *blob, int tensor_length);

void multiply_int16_elementwise_constant_ref(int16_t *output, int16_t *input1, void *blob, int tensor_length) {
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

void multiply_int16_elementwise_constant(int16_t *output, int16_t *input1, void *transformed_input2, int tensor_length) {
#ifdef NN_USE_REF
    multiply_int16_elementwise_constant_ref(output, input1, transformed_input2, tensor_length);
#else
    multiply_int16_elementwise_constant_asm(output, input1, transformed_input2, tensor_length);
#endif
}


