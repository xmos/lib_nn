#include <stdint.h>
#include "nn_operator.h"
#include "nn_op_helper.h"
#include "math.h"

void softmax_generate_exp_lut(int zero_point, float scale, float *lut) {
  for (int i = 0; i < 256; i++) {
    float real_val = (float)(i - zero_point) * scale;
    lut[i] = expf(real_val);
  }
}

void softmax_exp_sum(float *Y, const int8_t X[], const float *lut,
                     const unsigned elm_start, const unsigned elm_count) {
  float sum = 0.0f;
  for (int i = elm_start; i < elm_start + elm_count; i++) {
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
  for (int i = elm_start; i < elm_start + elm_count; i++) {
    Y[i] = (int8_t)(lut[X[i] + 128] * inv_sum - 128.5f);
  }
}
