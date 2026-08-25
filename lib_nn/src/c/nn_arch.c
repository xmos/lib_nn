// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "nn_arch.h"

nn_target_arch_t NN_ARCH = TARGET_ARCH_XS3A;

void SetNNTargetArch(nn_target_arch_t arch) {
    NN_ARCH = arch;
}