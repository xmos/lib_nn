#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "quantize_int16_transform.h"

#define SHIFT  14

int quantize_int16_tensor_blob(void *output,
                                 float input_scaler) {
    float *blob = (float *)output;
    input_scaler = ldexp(input_scaler, 23);
    blob[0] = 1/input_scaler;
    return 1;
}
