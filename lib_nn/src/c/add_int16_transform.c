#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "add_int16_transform.h"

#define SHIFT  14

// Calculate the ratio with which each input should be scaled
// Shift those up by 14 bits, the amount by which the run time code shifts it back
// Fail if it doesn't fit in 16 bits
// Pass and broadcast it to two vectors (one for each input) if it does fit.
int add_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler,
                               char *err_msg) {
    int tensor_length = 16;
    int16_t *output_tensor = (int16_t *) output;
    float combined_scaler1 = input1_scaler / output_scaler;
    float combined_scaler2 = input2_scaler / output_scaler;
    int mult1 = round(combined_scaler1 * (1 << SHIFT));
    int mult2 = round(combined_scaler2 * (1 << SHIFT));
    if (mult1 > 32767 || mult2 > 32767 || mult1 < -32768 || mult2 < -32768) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                "Add FAIL! Input1 scaler is %g, input2 scaler is %g, and output scaler is %g",
                input1_scaler, input2_scaler, output_scaler);
        return 0;
    }
    for(int i = 0; i < tensor_length; i++) {
        output_tensor[i                ] = mult1;
        output_tensor[i + tensor_length] = mult2;
    }
    return 1;
}
