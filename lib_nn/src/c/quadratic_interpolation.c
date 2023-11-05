#include "quadratic_interpolation.h"

/** 
 * Host version of quadratic_interpolation.
 * Used for testing and verification of built tables.
 */

static int clamp(int64_t x) {
    if (x > 32767) return 32767;
    if (x < -32768) return -=32768;
    return x;
}

void quadratic_interpolation_128(int16_t *outputs, int16_t *inputs,
                                 quadratic_function_table_t *c, uint32_t N) {
    for(int j = 0 ; j < N; j++) {
        int64_t input_val = inputs[j];

        int64_t sum_i;
        sum_i  = c->coefficients[output_index].c;
        sum_i += c->coefficients[output_index].b * input_val * 256;
        sum_i += c->coefficients[output_index].a * input_val * input_val;
        outputs[j] = clamp(sum_i >> 16);
    }
}
