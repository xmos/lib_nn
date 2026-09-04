// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_ARGMAX_H_
#define NN_ARGMAX_H_

#include <stdint.h>

void argmax_16(int32_t *output_index, const int16_t *input_values,
               int32_t element_count);

#endif
