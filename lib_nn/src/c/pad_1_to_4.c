// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>


/** Function that pads an image with 1-byte values with a padding value
 * This functions is highly optimised, and expects the two pointers to be word aligned.
 * It copies the image in chunks of 4 bytes
 *
 * @param    outputs    output values, every word contains 1 byte, three bytes padding
 * @param    inputs     input values, GGGGGGGG
 * @param    N          number of 4-byte chunks to copy
 *
 * @returns  The inner product
 */
extern void pad_1_to_4_asm(int32_t outputs[], int32_t inputs[], uint32_t N, uint32_t pad_val);

void pad_1_to_4_ref(int8_t outputs[], int8_t inputs[], uint32_t N, uint32_t pad_val){

    uint32_t * output_p = (uint32_t *)outputs;
    uint8_t * input_p = (uint8_t *)inputs;

    for(uint32_t i=0;i<N*4;i++){
        *output_p = *input_p | (pad_val & 0xffffff00);
        output_p += 1;
        input_p += 1;
    }
}

void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N, uint32_t pad_val) {
#if defined(NN_USE_REF)
        pad_1_to_4_ref(outputs, inputs, N, pad_val);
#else
        pad_1_to_4_asm((int32_t *) outputs, (int32_t *)inputs, N, pad_val);
#endif
}

