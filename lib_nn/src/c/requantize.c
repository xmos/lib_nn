// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>

#include "multiply_int16.h"
#include "multiply_int16.h"
#include "nn_op_helper.h"

const int16_t eight_thousand[16] = {
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
  0x8000,
};

#if CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

#ifdef NN_USE_REF
void requantize_16_to_8_ref(int8_t *y, const int16_t *x,
                            const unsigned elm_start,
                            const unsigned elm_count) {
  for (unsigned i = elm_start; i < elm_start + elm_count; i++) {
    y[i] = (x[i] < -0x7F80) ? NEG_SAT_VAL : vdepth8_single_s16(x[i]);
  }
}

void requantize_16_to_8(int8_t *y, const int16_t *x, const unsigned elm_start,
                        const unsigned elm_count) {
  requantize_16_to_8_ref(y, x, elm_start, elm_count);
}
#endif // NN_USE_REF

#ifdef NN_USE_REF
void requantize_int16_tensor_ref(int16_t *output, int16_t *input1,
                                 int tensor_length, void *blob) {
  int16_t *multipliers = (int16_t *)blob;
  for (int i = 0; i < tensor_length; i++) {
    int64_t mult = (((int)input1[i]) << 16) +
                   input1[i] * (int64_t)multipliers[i & 15] * 2;
    mult = mult + (1 << 15);
    mult = mult >> 16;

    if (mult > 32767) mult = 32767;
    if (mult < -32768) mult = -32768;
    output[i] = mult;
  }
}

#else

extern void requantize_int16_tensor_asm(int16_t *output, int16_t *input1,
                                        int tensor_length, void *blob);

#endif

void requantize_int16_tensor(int16_t *output, int16_t *input1,
                             int tensor_length, void *blob) {
#ifdef NN_USE_REF
  requantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
  requantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}

#undef NEG_SAT_VAL
