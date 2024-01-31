#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "dequantize_int16_transform.h"

#define SHIFT  14

int dequantize_int16_tensor_blob(void *output,
                                 float input_scaler) {
    float *blob = (float *)output;
    blob[0] = ldexpf(input_scaler, 128+8);
    if (isinf(blob[0])) {
        return 0;
    }
    printf("%08x\n", ((int *)blob)[0]);
    blob[1] = input_scaler * -0x8000;
    return 1;
}
