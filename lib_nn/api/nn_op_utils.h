// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_OP_UTILS_H_
#define NN_OP_UTILS_H_

#include <stdint.h>
#include <string.h>

#include "nn_api.h"

C_API int calculateAlignedThreadSplit(int tc, int split_size, int split_start[], int split_end[]);
C_API int calculateThreadSplit(int tc, int split_size, int split_start[], int split_end[], int alignment);

#endif // NN_OP_UTILS_H_
