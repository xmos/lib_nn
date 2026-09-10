// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>

#include "nn_layers.h"

void pad_3_to_4_prepare(uint32_t * n_3,
    const unsigned height,
    const unsigned width) {
    *n_3 = height*width;
}

#if NN_USE_REF

void pad_1_to_4_ref(int8_t outputs[], int8_t inputs[], uint32_t N, uint32_t pad_val){

    uint32_t * output_p = (uint32_t *)outputs;
    uint8_t * input_p = (uint8_t *)inputs;

    for(uint32_t i=0;i<N*4;i++){
        *output_p = *input_p | (pad_val & 0xffffff00);
        output_p += 1;
        input_p += 1;
    }
}
void pad_3_to_4_ref(int8_t outputs[], int8_t inputs[], uint32_t N_3, uint32_t pad_val){

    int8_t * output_p = (int8_t *)outputs;
    int8_t * input_p = (int8_t *)inputs;

    for(uint32_t i=0;i<N_3;i++){
        memcpy(output_p, input_p, 3);
        output_p += 3;
        input_p += 3;
        *output_p = (int8_t)(pad_val >> 24);
        output_p += 1;
    }
}
#else

extern void pad_1_to_4_asm(int32_t outputs[], int32_t inputs[], uint32_t N,
                            uint32_t pad_val);
extern void pad_3_to_4_asm(int32_t outputs[], int64_t inputs[], uint32_t N_24,
                            uint32_t pad_val);

static inline void pad_3_to_4_single(int8_t **outputs, int8_t **inputs, uint32_t *N_3, uint32_t pad_val) {
    for(uint32_t i = 0; i < 3; i++) {
        (*(int8_t**)outputs)[i] = (*inputs)[i];
    }
    (*(int8_t**)outputs)[3] = (int8_t)(pad_val >> 24);
    *inputs += 3;
    *outputs += 4;
    *N_3 -= 1;
}

void pad_3_to_4_run_impl(int8_t outputs[], int8_t inputs[], uint32_t N_3, uint32_t pad_val) {
    // First copy single pixels until the input pointer is aligned
    // That will happen as it is incremented in steps of 3
    // But we may run out of pixels before it happens
    while((((uint32_t)inputs) & 7) != 0 && N_3 != 0) {
        pad_3_to_4_single(&outputs, &inputs, &N_3, pad_val);
    }

    // Now figure out whether the total number of pixels to be copied
    // Is a multiple of 24; if not, remember what the remainder is
    uint32_t tail_N_3 = N_3 & 7;    // remaining blocks of 3
    uint32_t N_24 = N_3 >> 3;       // Blocks of 24

    // Now copy the bulk of the data in blocks of 24
    if (N_24 != 0) {

        pad_3_to_4_asm((int32_t * )outputs, (int64_t *)inputs, N_24, pad_val);
    }

    // Finally, if there is a remainder, copy them a pixel at a time
    if (tail_N_3 != 0) {
        // Adjust the inputs and outputs pointer to point to the remainder.
        inputs +=  (N_24 << 3) * 3;
        outputs += (N_24 << 3) * 4;
        while(tail_N_3 != 0) {
            pad_3_to_4_single(&outputs, &inputs, &tail_N_3, pad_val);
        }
    }
}
#endif

void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N, uint32_t pad_val) {
#if NN_USE_REF
    pad_1_to_4_ref(outputs, inputs, N, pad_val);
#else
    pad_1_to_4_asm((int32_t *)outputs, (int32_t *)inputs, N, pad_val);
#endif
}

void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3, uint32_t pad_val) {
#if NN_USE_REF
    pad_3_to_4_ref(outputs, inputs, N_3, pad_val);
#else
    pad_3_to_4_run_impl(outputs, inputs, N_3, pad_val);
#endif
}
