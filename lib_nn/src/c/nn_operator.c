// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "nn_operator.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../nn_op_helper.h"
#include "xs3_vpu.h"

// Note: There currently is no assembly implementation.
void argmax_16(int32_t* Y, const int16_t* X, const int32_t N) {
  if (N <= 0) return;

  *Y = 0;

  for (int32_t i = 1; i < N; i++) {
    if (X[i] > X[*Y]) {
      *Y = i;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

// static void requantize_16_to_8_prepare(
//     nn_requantize_16_to_8_job_t* jobs,
//     const uint32_t length,
//     unsigned job_count)
// {

//     for(int k = 0; k < job_count; k++){
//         jobs[k].length = 0;
//     }

//     int32_t left = (length >> 2) << 2;

//     while(left){
//         for(int k = 0; k < job_count; k++){
//             if(left >= 4){
//                 jobs[k].length += 4;
//                 left -= 4;
//             } else {
//                 jobs[k].length += left;
//                 left -= left;
//             }
//         }
//         if(left == 0) break;
//     }
//     jobs[job_count-1].length += (length % 4);

//     jobs[0].start = 0;

//     int32_t pos = jobs[0].length;

//     for(int k = 1; k < job_count; k++){
//         jobs[k].start = jobs[k-1].start + jobs[k-1].length;
//         pos += jobs[k].length;
//     }

//     assert(pos == length);
// }

#if CONFIG_SYMMETRIC_SATURATION_requantize_16_to_8
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

void requantize_16_to_8_ref(int8_t* y, const int16_t* x,
                            const unsigned elm_start,
                            const unsigned elm_count) {
  for (int i = elm_start; i < elm_start + elm_count; i++) {
    y[i] = (x[i] < -0x7F80) ? NEG_SAT_VAL : vdepth8_single_s16(x[i]);
  }
}

#undef NEG_SAT_VAL

///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////

void lookup8_ref(uint8_t* Y, const uint8_t* X, const uint8_t* lut,
                 const unsigned elm_start, const unsigned elm_count) {
  for (int i = elm_start; i < elm_start + elm_count; i++) {
    Y[i] = lut[X[i]];
  }
}

#ifdef NN_USE_REF

void requantize_16_to_8(int8_t* y, const int16_t* x, const unsigned elm_start,
                        const unsigned elm_count) {
  requantize_16_to_8_ref(y, x, elm_start, elm_count);
}

void lookup8(uint8_t* Y, const uint8_t* X, const uint8_t* lut,
             const unsigned elm_start, const unsigned elm_count) {
  lookup8_ref(Y, X, lut, elm_start, elm_count);
}

#endif  // NN_USE_REF