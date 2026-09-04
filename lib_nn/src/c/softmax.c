// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "math.h"
#include "nn_op_helper.h"
#include "softmax.h"
#include <stdint.h>

static int clamp8(double x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return x;
}

void softmax_generate_exp_lut(int zero_point, float scale, float *lut) {
  for (int i = 0; i < 256; i++) {
    float real_val = (float)(i - zero_point) * scale;
    lut[i] = expf(real_val);
  }
}

void softmax_exp_sum(float *Y, const int8_t X[], const float *lut,
                     const unsigned elm_start, const unsigned elm_count) {
  float sum = 0.0f;
  for (unsigned i = elm_start; i < elm_start + elm_count; i++) {
    sum += lut[X[i] + 128];
  }
  *Y = sum;
}

void softmax_calculate_inv_sum(float *inv_sum, const float sums[]) {
  *inv_sum = 1.0f / (sums[0] + sums[1] + sums[2] + sums[3] + sums[4]) * 256.0f;
}

// Assumes overflows can't occur because of quantization: check this in
// compiler!!
void softmax_exp_div(int8_t Y[], const int8_t X[], const float *lut,
                     const float inv_sum, const unsigned elm_start,
                     const unsigned elm_count) {
  for (unsigned i = elm_start; i < elm_start + elm_count; i++) {
    Y[i] = (int8_t)clamp8(roundf(lut[X[i] + 128] * inv_sum) - 128);
  }
}

void softmax_single(int8_t Y[], const int8_t X[], const float *lut,
                    const int offset) {
  float sum = 0.0f;
  for (int i = 0; i < offset; i++) {
    sum += lut[X[i] + 128];
  }
  const float inv_sum = 1.0f / sum * 256.0f;
  for (int i = 0; i < offset; i++) {
    Y[i] = (int8_t)clamp8(roundf(lut[X[i] + 128] * inv_sum) - 128);
  }
}

#ifdef NN_USE_REF
// Reference implementation: as accurate as possible
// Round to int before casting
// Minus max float value to avoid numerical instability:
// exp(arr) / sum(exp(arr)) = exp(arr + C) / sum(exp(arr + C))
void softmax_ref(int8_t *Y, const int8_t *X, const float zero_point,
                 const float scale, const int length) {
  int8_t max_val = X[0];
  for (int i = 1; i < length; i++) {
    max_val = X[i] > max_val ? X[i] : max_val;
  }
  const float max_val_f = ((float)max_val - zero_point) * scale;
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += expf(((float)X[i] - zero_point) * scale - max_val_f);
  }
  for (int i = 0; i < length; i++) {
    const float real_val =
        (expf(((float)X[i] - zero_point) * scale - max_val_f) / sum);
    Y[i] = (int8_t)(real_val * 256 - 128.5f);
  }
}
#endif // NN_USE_REF

void softmax(int8_t *Y, const int8_t *X, const float zero_point,
             const float scale, const int length) {
#ifdef NN_USE_REF
  softmax_ref(Y, X, zero_point, scale, length);
#else
  float lut[256];
  float sums[5] = {0.0f};
  float inv_sum;

  softmax_generate_exp_lut((int)zero_point, scale, lut);
  softmax_exp_sum(&sums[0], X, lut, 0, (unsigned)length);
  softmax_calculate_inv_sum(&inv_sum, sums);
  softmax_exp_div(Y, X, lut, inv_sum, 0, (unsigned)length);
#endif // NN_USE_REF
}
