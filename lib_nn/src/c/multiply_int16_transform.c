#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "multiply_int16_transform.h"

int multiply_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler) {
    int tensor_length = 16;
    int16_t *output_tensor = (int16_t *) output;
    float combined_scaler = input1_scaler * input2_scaler / output_scaler;
    assert(combined_scaler > 0);
    int shift = floor(log2(32767 / combined_scaler));
    int mult = ldexp(combined_scaler, shift);
    if (mult > 32767) {
        return 0;
    }
    for(int i = 0; i < tensor_length; i++) {
        output_tensor[i] = mult;
    }
    output_tensor[16] = shift;
    return 1;
}

int requantize_int16_tensor_blob(void *blob,
                                 float input1_scaler,
                                 float output_scaler) {
    int16_t *output_tensor = (int16_t *) blob;
    int tensor_length = 16;
    float combined_scaler = input1_scaler / output_scaler;
    assert(combined_scaler > 0);
    int mult = round((combined_scaler-1) * 32768);
    if (mult > 32767 || mult < -32768) {
        return 0;
    }
    for(int i = 0; i < tensor_length; i++) {
        output_tensor[i] = mult;
    }
    return 1;
}
