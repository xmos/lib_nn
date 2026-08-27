// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__) || defined(__VX4B__)
#include "nn_operator.h"
#include "../src/asm/asm_constants.h"
#include "vpu_sim.h"

void dequantize_int16_tensor_ref(float *output, int16_t *input, int tensor_length, void *blob);
void dequantize_int16_tensor_asm(float *output, int16_t *input, int tensor_length, void *blob) {
    dequantize_int16_tensor_ref(output, input, tensor_length, blob);
}
void multiply_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) ;
void multiply_int16_tensor_asm(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    multiply_int16_tensor_ref(output, input1, input2, tensor_length, blob);
}

void requantize_int16_tensor_ref(int16_t *output, int16_t *input1, int tensor_length, void *blob) ;
void requantize_int16_tensor_asm(int16_t *output, int16_t *input1, int tensor_length, void *blob) {
    requantize_int16_tensor_ref(output, input1, tensor_length, blob);
}

void pad_3_to_4_asm(int32_t outputs[], int64_t inputs[], uint32_t N_24, uint32_t pad_val) {
    int8_t * outputs_p = (int8_t *)outputs;
    int8_t * inputs_p = (int8_t *)inputs;
    for(uint32_t l=0;l<N_24;l++){
        for (unsigned i=0;i<8;i++){
            memcpy(outputs_p, inputs_p, 3);
            inputs_p += 3;
            outputs_p += 3;
            memcpy(outputs_p, &pad_val, 1);
            outputs_p += 1;
        }
    }
};

int16_t *output_transform_fn_int16_impl(int16_t *vDvR,
                                        int32_t *mul_add,
                                        int16_t *output,
                                        uint32_t N);
void output_transform_fn_int16_impl_asm(int16_t *vDvR,
                                        int32_t *mul_add,
                                        int16_t *output,
                                        uint32_t N) {
    output_transform_fn_int16_impl(vDvR, mul_add, output, N);
}

void quantize_int16_tensor_ref(int16_t *output, float *input, int tensor_length, void *blob);
void quantize_int16_tensor_asm(int16_t *output,
                               float *input, int tensor_length, void *blob) {
    quantize_int16_tensor_ref(output, input, tensor_length, blob);
}

// This empty stub here is to pass the build, this will not be called
int8_t *output_transform_fn_int_channelwise_impl_asm(
    const void *params, int8_t *Y, void *A,
    int16_t *multipliers_and_biases, int output_count) { return NULL; }

#endif
