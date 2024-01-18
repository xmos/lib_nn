#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>

void multiply_int16_transform(void *output,
                              int16_t *constant_multiplier_tensor,
                              int tensor_length,
                              float input1_scaler,
                              float input2_scaler,
                              float output_scaler) {
    int32_t *output_step = (int32_t *) output;
    *output_step = 32;
    int16_t *output_tensor = (int16_t *) &output_step[1];
    float combined_scaler = input1_scaler * input2_scaler / output_scaler;
    assert(combined_scaler > 0);
    for(int i = 0; i < tensor_length; i++) {
        int lower_i  = i & 15;
        int higher_i = i & ~15;
        int transform_index = higher_i << 1 | lower_i;
        int shift_index     = transform_index + 16;
        int exponent = 0;
        float mod = constant_multiplier_tensor[i] * combined_scaler;
        if (mod != 0) {
            if (mod > 0) {
                while(round(mod) < 16384) {
                    mod *= 2;
                    exponent++;
                }
            } else {
                while(round(mod) >= -16384) {
                    mod *= 2;
                    exponent++;
                }
            }
        }
        assert(round(mod) < 32768 && round(mod) >= -32768);
        output_tensor[transform_index] = round(mod);
        output_tensor[shift_index] = exponent;
    }
}

void requantise_int16_transform(void *output,
                                float input1_scaler,
                                float output_scaler) {
    int32_t *output_step = (int32_t *) output;
    *output_step = -32;
    int16_t *output_tensor = (int16_t *) &output_step[1];
    int constant = 1;
    float input2_scaler = 1.0;
    float combined_scaler = input1_scaler * input2_scaler / output_scaler;
    assert(combined_scaler > 0);
    float mod = constant * combined_scaler;
    assert(mod > 0);
    int exponent = 0;
    while(round(mod) < 16384) {
        mod *= 2;
        exponent++;
    }
    assert(round(mod) < 32768 && round(mod) >= -32768);
    for(int i = 0; i < 16; i++) {
        int lower_i  = i & 15;
        int higher_i = i & ~15;
        int transform_index = higher_i << 1 | lower_i;
        int shift_index     = transform_index + 16;
        output_tensor[transform_index] = round(mod);
        output_tensor[shift_index] = exponent;
    }
}
