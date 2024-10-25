// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "nn_operator.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "nn_op_helper.h"
#include "xs3_vpu.h"

int calculateThreadSplit(int tc, int split_size, int split_start[],
                          int split_end[], int alignment) {
  split_start[0] = 0;

  // Figure out min number of threads needed while keeping alignment
  int threads_needed = (split_size + (alignment - 1)) / alignment;
  tc = tc > threads_needed ? threads_needed : tc;

  for (int i = 0; i < tc; i++) {
    int split = (split_size + (tc - i) - 1) / (tc - i);
    split_size -= split;
    if (split > 0) {
      split_end[i] = split_start[i] + split;
      if (i != tc - 1)
        split_start[i + 1] = split_end[i];
    } else {
      break;
    }
  }

  // Align up or down split_starts,
  // so that each thread begins work at an aligned address
  // The last thread handles remaining items, so don't modify the end
  for (int i = 1; i < tc; i++) {
    if ((split_start[i] & (alignment - 1)) > (alignment >> 2)) {
      // Align up
      split_start[i] = (split_start[i] + (alignment - 1)) & ~(alignment - 1);
    } else {
      // Align down
      split_start[i] = split_start[i] & ~(alignment - 1);
    }
    split_end[i - 1] = split_start[i];
  }

  return tc;
}

int calculateAlignedThreadSplit(int tc, int split_size, int split_start[],
                          int split_end[]) {
return calculateThreadSplit(tc, split_size, split_start,
                          split_end, /*alignment=*/4);
}

// Note: There currently is no assembly implementation.
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

void requantize_16_to_8_ref(int8_t *y, const int16_t *x,
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

void lookup8_ref(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
                 const unsigned elm_start, const unsigned elm_count) {
  for (int i = elm_start; i < elm_start + elm_count; i++) {
    Y[i] = lut[X[i]];
  }
}

#ifdef NN_USE_REF
void requantize_16_to_8(int8_t *y, const int16_t *x, const unsigned elm_start,
                        const unsigned elm_count) {
  requantize_16_to_8_ref(y, x, elm_start, elm_count);
}
#endif // NN_USE_REF

void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
             const unsigned elm_start, const unsigned elm_count) {
#ifdef NN_USE_REF
  lookup8_ref(Y, X, lut, elm_start, elm_count);
#else
  lookup8_asm(Y, X, lut, elm_start, elm_count);
#endif // NN_USE_REF
}

// Reference implementation: as accurate as possible
// Round to int before casting
// Minus max float value to avoid numerical instability:
// exp(arr) / sum(exp(arr)) = exp(arr + C) / sum(exp(arr + C))
void softmax_ref(int8_t *Y, const int8_t *X, const float zero_point,
                 const float scale, const int length) {
  int8_t max_val = X[0];
  for (int i = 1; i < length; i++) {
    max_val = X[i] > max_val ? X[i] : max_val;
  }
  const float max_val_f = ((float)max_val - zero_point) * scale;
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += expf(((float)X[i] - zero_point) * scale - max_val_f);
  }
  for (int i = 0; i < length; i++) {
    const float real_val =
        (expf(((float)X[i] - zero_point) * scale - max_val_f) / sum);
    Y[i] = (int8_t)(real_val * 256 - 128.5f);
  }
}
