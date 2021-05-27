// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <syscall.h>

#include "meas_common.h"
#include "nn_operator.h"
#include "xs3_vpu.h"

void benchmark_requantize_16_to_8(int argc, char** argv) {
  assert(argc >= 1);

  for (int k = 0; k < argc; k++) {
    unsigned input_count = atoi((char*)argv[k]);

    int16_t* src = malloc(input_count * sizeof(int16_t));
    int8_t* dst = malloc(input_count * sizeof(int8_t));

    assert(src);
    assert(dst);

    requantize_16_to_8(dst, src, input_count);

    free(src);
    free(dst);
  }
}