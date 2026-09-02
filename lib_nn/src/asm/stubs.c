// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__) || defined(__VX4B__)
#include "nn_operator.h"
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

void quantize_int16_tensor_ref(int16_t *output, float *input, int tensor_length, void *blob);
void quantize_int16_tensor_asm(int16_t *output,
                               float *input, int tensor_length, void *blob) {
    quantize_int16_tensor_ref(output, input, tensor_length, blob);
}

#endif
