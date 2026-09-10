// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

#include "nn_arch.h"
#include "nn_layers.h"

static inline 
unsigned get_shift(void){
    unsigned shift = NN_ARCH == TARGET_ARCH_XS3A ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
    return shift;
}

int add_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler,
                               char *err_msg) {
    const unsigned shift = get_shift();
    int tensor_length = 16;
    int16_t *output_tensor = (int16_t *) output;
    float combined_scaler1 = input1_scaler / output_scaler;
    float combined_scaler2 = input2_scaler / output_scaler;
    int mult1 = round(combined_scaler1 * (1 << shift));
    int mult2 = round(combined_scaler2 * (1 << shift));
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

#if NN_USE_REF
void add_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    int16_t *multipliers = (int16_t *) blob;
    const unsigned shift = get_shift();
    for(int i = 0; i < tensor_length; i++) {
        int64_t mult1 = input1[i] * (int64_t) multipliers[(i & 15)     ];
        int64_t mult2 = input2[i] * (int64_t) multipliers[(i & 15) + 16];
        int answer = (mult1 + mult2 + (1 << (shift - 1))) >> shift;

        if (answer > 32767) answer = 32767;
        if (answer < -32768) answer = -32768;
        output[i] = answer;
    }
}
#else
extern void add_int16_tensor_asm(
    int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob
);
#endif


void add_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    add_int16_tensor_ref(output, input1, input2, tensor_length, blob);
#else
    add_int16_tensor_asm(output, input1, input2, tensor_length, blob);
#endif
}
