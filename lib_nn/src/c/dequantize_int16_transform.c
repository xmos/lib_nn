#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dequantize_int16_transform.h"

#define SHIFT  14

int dequantize_int16_tensor_blob(void *output,
                                 float input_scaler,
                                 char *err_msg) {
    float *blob = (float *)output;
    blob[0] = ldexpf(input_scaler, 128+8);
    if (isinf(blob[0])) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                "Dequantize FAIL! Input scaler is %g",
                input_scaler);
        return 0;
    }
    blob[1] = input_scaler * -0x8000;
    return 1;
}
