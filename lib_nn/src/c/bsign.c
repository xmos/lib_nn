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

#if defined(__XS3A__) && !defined(NN_USE_REF)
/* Bsign_8 optimised for XS3A */
void bsign_8_( 
    bnn_b32_t* y,
    const int8_t* x,
    const int8_t* zero_point_vect,
    const nn_bsign_8_job_t* job);

void bsign_8(
    bnn_b32_t* y,
    const int8_t* x,
    const nn_bsign_8_plan_t * plan,
    const nn_bsign_8_job_t* job)
{
    /* This implementation follows the convention of existing code - we are passing around a plan which contains 
     * 1 byte zero_point. This actually costs us more in terms of memory usage than simply passing around the value.
     * Worse than this we then generate a zero point vector PER JOB effecting memory and runtime 
     * Therefore, TODO, put the zero-point vector in the plan and generate in init() 
     */
    int8_t zero_point_vect[VPU_INT8_EPV];
    memset(zero_point_vect, plan->zero_point, sizeof(zero_point_vect));

    /* Note, at this point we have no more use for the plan..*/
    bsign_8_(y, x, (const int8_t*)&zero_point_vect, job);
}
#endif // defined(__XS3A__) && !defined(NN_USE_REF)

void bsign_8_prepare(
    nn_bsign_8_plan_t* plan,
    nn_bsign_8_job_t* jobs,
    const uint32_t length,
    const int8_t zero_point,
    unsigned job_count)
{
    plan->zero_point = zero_point;

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
