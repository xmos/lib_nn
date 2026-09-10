// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdint.h>

#include "nn_layers.h"

void argmax_16(int32_t *Y, const int16_t *X, const int32_t N) {
  if (N <= 0)
    return;

  *Y = 0;

  for (int32_t i = 1; i < N; i++) {
    if (X[i] > X[*Y]) {
      *Y = i;
    }
  }
}