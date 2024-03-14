#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "multiply_int16_transform.h"

/*
 * Representative example:
 *    input1 scaler = 0.000578337523621,
 *    input2 scaler = 0.000020991212295,
 *    output scaler = 0.000093780785392.
 *      Multiply numbers [-19.0..19.0) with [-0.68..0.68) into [-3.0..3.0)
 *      (a lot of clipping)
 *      combined_scaler = 0.0001294508858
 *      log2 = 27.95
 *      shift = 27
 *      multiplier = 17374(.60377)
 *      at run time calculate A * B * 17374 >> 27
 *                            3458 (2.0) * 23819 (0.5) * 17374 = 10662 (0.9998) 
 */
int multiply_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler,
                               char *err_msg) {
    int16_t *output_tensor = (int16_t *) output;
    float combined_scaler = input1_scaler * input2_scaler / output_scaler;
    assert(combined_scaler > 0);
    int shift = floor(log2(32768 / combined_scaler));
    int mult = ldexp(combined_scaler, shift);
    if (mult > 32767) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                "Mul FAIL! Input1 scaler is %g, input2 scaler is %g, and output scaler is %g",
                input1_scaler, input2_scaler, output_scaler);
        return 0;
    }
    output_tensor[0] = mult;
    output_tensor[1] = shift;
    return 1;
}

int requantize_int16_tensor_blob(void *blob,
                                 float input1_scaler,
                                 float output_scaler,
                                 char *err_msg) {
    int16_t *output_tensor = (int16_t *) blob;
    int tensor_length = 16;
    float combined_scaler = input1_scaler / output_scaler;
    assert(combined_scaler > 0);
    int mult = round((combined_scaler-1) * 32768);
    if (mult > 32767 || mult < -32768) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                "Requantize FAIL! Input scaler is %g and output scaler is %g",
                input1_scaler, output_scaler);
        return 0;
    }
    for(int i = 0; i < tensor_length; i++) {
        output_tensor[i] = mult;
    }
    return 1;
}
