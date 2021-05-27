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

uint8_t LUT[256] = {0};

void benchmark_lookup8(int argc, char** argv) {
  assert(argc >= 1);

  for (int k = 0; k < argc; k++) {
    unsigned input_count = atoi((char*)argv[k]);

    uint8_t* buff = malloc(input_count * sizeof(int8_t));

    assert(buff);

    lookup8(buff, buff, LUT, input_count);

    free(buff);
  }
}