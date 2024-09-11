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
    blob[0] = input_scaler * (1<<22);
    if (isinf(blob[0])) {
        snprintf(err_msg, ERR_MSG_DESCRIPTOR_FAIL_BYTES(),
                "Dequantize FAIL! Input scaler is %g",
                input_scaler);
        return 0;
    }
    ((int *)blob)[1] = 0xc0008000;         // This is -0x40008000 which is added during conversion
    return 1;
}
