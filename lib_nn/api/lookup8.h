// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_LOOKUP8_H_
#define NN_LOOKUP8_H_

#include <stdint.h>

void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
             unsigned elm_start, unsigned elm_count);

#endif
