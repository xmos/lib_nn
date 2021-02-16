// Copyright 2020 XMOS LIMITED. This Software is subject to the terms of the 
// XMOS Public License: Version 1
#include "nn_operator.h"
#include "../nn_op_helper.h"

#include "xs3_vpu.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

void bsign_8_prepare(
    nn_bsign_8_job_t* jobs,
    int8_t* zero_point_vect,
    const uint32_t length,
    const int8_t zero_point,
    unsigned job_count)
{
    memset(zero_point_vect, zero_point, VPU_INT8_EPV * sizeof(zero_point));

    // decompose length = k * job_count * 32 + p * 32 + r,
    // where 0 <= p < job_count and 0 <= r < 32
    int32_t r = length % 32;
    int32_t full_vectors = (length - r) / 32;
    int32_t p = full_vectors % job_count;
    int32_t k = (full_vectors - p) / job_count;
    assert(length == k * job_count * 32 + p * 32 + r);

    for(int j = 0; j < job_count; j++)
    {
        jobs[j].start = (j > 0) ? jobs[j-1].start + jobs[j-1].length : 0;
        jobs[j].length = k * 32 + (j < p) * 32;
    }
    jobs[job_count-1].length += r;
}

void bsign_8_ref(
    bnn_b32_t* y,
    const int8_t* x,
    const int8_t* zero_point_vect,
    const nn_bsign_8_job_t* job)
{
    y = ADDR(y, job->start/32);
    x = ADDR(x, job->start);

    for(int input_idx = 0; input_idx < job->length; input_idx++)
    {
        int32_t output_idx = input_idx / 32;
        int32_t shift = input_idx % 32;

        // Note, this matches Larq - where 0's are witten to the upper unused bytes of the tail word
        if(shift == 0)
            y[output_idx] = 0;

        if(x[input_idx] < zero_point_vect[shift])
            y[output_idx] |= (1 << shift);
    }
}

#ifdef NN_USE_REF
void bsign_8( 
    bnn_b32_t* y,
    const int8_t* x,
    const int8_t* zero_point_vect,
    const nn_bsign_8_job_t* job)
{
    bsign_8_ref(y, x, zero_point_vect, job);
}
#endif // NN_USE_REF

