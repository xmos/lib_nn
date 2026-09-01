// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "quantize_int16_transform.h"

int quantize_int16_tensor_blob(void *output,
                                 float output_scaler) {
    float *blob = (float *)output;
    output_scaler = ldexp(output_scaler, 23);
    blob[0] = 1/output_scaler;
    return 1;
}
