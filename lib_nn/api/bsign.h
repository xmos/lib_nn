// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_BSIGN_H_
#define NN_BSIGN_H_

#include <stdint.h>
#include "nn_bin_types.h"
#include "nn_api.h"
#include "nn_image.h"

typedef struct {
    mem_stride_t start;
    int32_t length;
} nn_bsign_8_job_t;

void bsign_8_prepare(nn_bsign_8_job_t *jobs, int8_t *zero_point_vect,
                     const uint32_t N, const int8_t zero_point,
                     const int32_t job_count);
void bsign_8(bnn_b32_t *Y, const int8_t *X, const int8_t *zero_point_vect,
             const nn_bsign_8_job_t *job);

#endif
