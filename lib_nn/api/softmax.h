// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_SOFTMAX_H_
#define NN_SOFTMAX_H_

#include <stdint.h>

void softmax(int8_t *Y, const int8_t *X, float zero_point,
             float scale, int length);
void softmax_generate_exp_lut(int zero_point, float scale, float *lut);
void softmax_exp_sum(float *Y, const int8_t *X, const float *lut,
                     unsigned elm_start, unsigned elm_count);
void softmax_exp_div(int8_t *Y, const int8_t *X, const float *lut,
                     float inv_sum, unsigned elm_start, unsigned elm_count);
void softmax_calculate_inv_sum(float *inv_sum, const float sums[]);
void softmax_single(int8_t *Y, const int8_t *X, const float *lut, int offset);

#endif
