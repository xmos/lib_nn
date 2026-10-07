// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "nn_arch.h"

#if VPU_CONFIGURED
nn_target_arch_t NN_ARCH = VPU_VLMUL_SHIFT_OFFSET == 2
                               ? TARGET_ARCH_XS3A
                               : TARGET_ARCH_VX4A;
nn_vpu_config_t NN_VPU_CONFIG = {VPU_SYMMETRIC_SATURATION,
                                 VPU_VLMACC_PRODUCT_LSB_DROPS,
                                 VPU_VLMUL_SHIFT_OFFSET};
#else
nn_target_arch_t NN_ARCH = TARGET_ARCH_XS3A;
#if defined(NN_USE_REF)
nn_vpu_config_t NN_VPU_CONFIG = {0, 0, 2};
#else
nn_vpu_config_t NN_VPU_CONFIG = {1, 0, 2};
#endif
#endif

void SetNNVPUConfig(nn_vpu_config_t config) {
    NN_VPU_CONFIG = config;
}

void SetNNTargetArch(nn_target_arch_t arch) {
    NN_ARCH = arch;
    if (arch == TARGET_ARCH_XS3A) {
        SetNNVPUConfig((nn_vpu_config_t){1, 0, 2});
    } else {
        SetNNVPUConfig((nn_vpu_config_t){0, 1, 1});
    }
}