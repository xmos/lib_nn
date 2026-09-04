// Copyright 2020-2026 XMOS LIMITED.
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
void argmax_16(int32_t *output_index, const int16_t *input_values,
               const int32_t element_count) {
  if (element_count <= 0)
    return;

  *output_index = 0;

  for (int32_t i = 1; i < element_count; i++) {
    if (input_values[i] > input_values[*output_index]) {
      *output_index = i;
    }
  }
}
