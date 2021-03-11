// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../nn_op_helper.h"
#include "nn_operator.h"
#include "xs3_vpu.h"

void bsign_8_prepare(nn_bsign_8_job_t* jobs, int8_t* zero_point_vect,
                     const uint32_t length, const int8_t zero_point,
                     const int32_t job_count) {
  memset(zero_point_vect, zero_point, VPU_INT8_EPV * sizeof(zero_point));

  // decompose length = k * job_count * VPU_INT8_EPV + p * VPU_INT8_EPV + r,
  // where 0 <= p < job_count and 0 <= r < VPU_INT8_EPV
  int32_t r = length % VPU_INT8_EPV;
  int32_t full_vectors = (length - r) / VPU_INT8_EPV;
  int32_t p = full_vectors % job_count;
  int32_t k = (full_vectors - p) / job_count;
  assert(length == k * job_count * VPU_INT8_EPV + p * VPU_INT8_EPV + r);

  for (int j = 0; j < job_count; j++) {
    jobs[j].start = (j > 0) ? jobs[j - 1].start + jobs[j - 1].length : 0;
    jobs[j].length = k * VPU_INT8_EPV + (j < p) * VPU_INT8_EPV;
  }
  jobs[job_count - 1].length += r;
}

void bsign_8_ref(bnn_b32_t* y, const int8_t* x, const int8_t* zero_point_vect,
                 const nn_bsign_8_job_t* job) {
  y = ADDR(y, job->start / VPU_INT8_EPV);
  x = ADDR(x, job->start);

  for (int input_idx = 0; input_idx < job->length; input_idx++) {
    int32_t output_idx = input_idx / VPU_INT8_EPV;
    int32_t shift = input_idx % VPU_INT8_EPV;

    // Note, this matches Larq - where 0's are witten to the upper unused bytes
    // of the tail word
    if (shift == 0) y[output_idx] = 0;

    if (x[input_idx] < zero_point_vect[shift]) y[output_idx] |= (1 << shift);
  }
}

#ifdef NN_USE_REF
void bsign_8(bnn_b32_t* y, const int8_t* x, const int8_t* zero_point_vect,
             const nn_bsign_8_job_t* job) {
  bsign_8_ref(y, x, zero_point_vect, job);
}
#endif  // NN_USE_REF
