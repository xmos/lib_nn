// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef HELPERS_H
#define HELPERS_H

#include "nn_operator.h"

extern const int post_vlmul_shr;

int pseudo_rand(int* seed);

int8_t saturate_output_int8(int value);

void pick_threshold_params(int32_t* thresholds, const unsigned chans_out,
                           const unsigned receptive_volume);

void pick_post_activation_params(float* post_activation_multiplier,
                                 float* post_activation_bias,
                                 unsigned chans_out, unsigned receptive_volume,
                                 int* seed);

void pick_extreme_bias_post_activation_params(float* post_activation_multiplier,
                                              float* post_activation_bias,
                                              unsigned chans_out,
                                              unsigned receptive_volume,
                                              int* seed);

void pick_extreme_mul_post_activation_params(float* post_activation_multiplier,
                                             float* post_activation_bias,
                                             unsigned chans_out,
                                             unsigned receptive_volume,
                                             int* seed);

#define DIV_BY_AND_ROUND_UP(x, y) (((x) + (y)-1) / (y))

#define max(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a > _b ? _a : _b;      \
  })

#define min(a, b)           \
  ({                        \
    __typeof__(a) _a = (a); \
    __typeof__(b) _b = (b); \
    _a < _b ? _a : _b;      \
  })

int clrsb(int x);

#endif  // HELPERS_H
