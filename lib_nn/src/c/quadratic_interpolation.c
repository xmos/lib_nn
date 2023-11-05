#include <assert.h>

#include "quadratic_interpolation.h"

/** 
 * Host version of quadratic_interpolation.
 * Used for testing and verification of built tables.
 */

static int clamp(int64_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -32768;
    return x;
}

void quadratic_interpolation_128(int16_t *outputs, int16_t *inputs,
                                 quadratic_function_table_t *c, uint32_t N) {
    int number_bytes = quadratic_function_table_number_bytes(c);
    assert(number_bytes = 128 * 8);
    uint8_t *bytes = quadratic_function_table_bytes(c);
    
    for(int j = 0 ; j < N; j++) {
        int64_t input_val = inputs[j];

        int table_index = (input_val >> 9) + (1 << 6);
        input_val = input_val &  ((1<<9) - 1) + (1<<8);
        int64_t sum_i;
        uint8_t *element = bytes + table_index * 8;
        int32_t c = *(int32_t*) element;
        int32_t b = *(int16_t*) (element + 4);
        int32_t a = *(int8_t*) (element + 7);
        sum_i  = c;
        sum_i += b * input_val * 256;
        sum_i += a * input_val * input_val;
        outputs[j] = clamp(sum_i >> 16);
    }
}
