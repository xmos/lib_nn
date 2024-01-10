#include <stdint.h>
#include "nn_operator.h"
#include "nn_op_helper.h"
#include "math.h"

void generateExpLUT(float zero_point, float scale, float *lut) {
  for (int i = 0; i < 256; i++) {
    int8_t quantized_val = i;
    float real_val = (quantized_val - zero_point) * scale;
    lut[i] = expf(real_val);
  }
}

void exp_sum(float *Y, const int8_t X[], const float *lut,
             const unsigned elm_start, const unsigned elm_count) {
  float sum = 0;
  for (int i = elm_start; i < elm_start + elm_count; i++) {
    sum += lut[X[i]];
  }
  *Y = sum;
}

void exp_div(int8_t Y[], const int8_t X[], const float *lut,
             const float inv_sum, const unsigned elm_start,
             const unsigned elm_count) {
  for (int i = elm_start; i < elm_start + elm_count; i++) {
    Y[i] = (int8_t)roundf(lut[X[i]] * inv_sum * 255 - 128);
  }
}
