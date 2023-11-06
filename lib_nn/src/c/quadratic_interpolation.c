#include <stdio.h>
#include <assert.h>

#include "quadratic_interpolation.h"

/** 
 * Host version of quadratic_interpolation.
 * Used for testing and verification of built tables.
 * The __xcore__ version will be picked up from quadratic_interpolation.S
 */

#ifndef __xcore__

static int clamp(int64_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return x;
}

void quadratic_interpolation_128(int16_t *outputs, int16_t *inputs,
                                 uint8_t *bytes, uint32_t N) {
    for(int j = 0 ; j < N; j++) {
        int64_t input_val = inputs[j];

        int table_index = (input_val >> 9) + (1 << 6);
        input_val = (input_val &  ((1<<9) - 1)) - (1<<8);
        int64_t sum_i;
        uint8_t *element = bytes + table_index * 8;
        int32_t c = *(int32_t*) element;
        int32_t b = *(int16_t*) (element + 6);
        int32_t a = *(int8_t*) (element + 4);
        sum_i  = c;
        sum_i += b * input_val * 256;
        sum_i += a * input_val * input_val;
        outputs[j] = clamp(sum_i >> 16);
    }
}

#endif 
