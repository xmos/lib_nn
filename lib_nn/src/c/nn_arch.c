// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "nn_arch.h"

// Default to the target being compiled for, so device code is correct without calling
// SetNNTargetArch(). Host builds default to XS3A and select the target at runtime.
#if defined(__riscv_xxcore)
nn_target_arch_t NN_ARCH = TARGET_ARCH_VX4A;
#else
nn_target_arch_t NN_ARCH = TARGET_ARCH_XS3A;
#endif

void SetNNTargetArch(nn_target_arch_t arch) {
    NN_ARCH = arch;
}