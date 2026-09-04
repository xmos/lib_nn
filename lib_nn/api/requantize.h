// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_REQUANTIZE_H_
#define NN_REQUANTIZE_H_

#include <stdint.h>
#include "nn_api.h"

void requantize_16_to_8(int8_t *y, const int16_t *x, unsigned elm_start,
                        unsigned elm_count);

#endif
