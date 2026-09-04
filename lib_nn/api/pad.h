// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_PAD_H_
#define NN_PAD_H_

#include <stdint.h>

void pad_3_to_4_prepare(uint32_t *n_3, unsigned height, unsigned width);
void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3,
                    uint32_t pad_val);
void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N,
                    uint32_t pad_val);

#endif
