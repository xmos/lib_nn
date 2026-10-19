// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "nn_arch.h"

#if defined(__riscv_xxcore)
nn_target_arch_t NN_ARCH = TARGET_ARCH_VX4A;
#else
nn_target_arch_t NN_ARCH = TARGET_ARCH_XS3A;
#endif

static nn_arch_profile_t NN_ARCH_PROFILES[TARGET_ARCH_COUNT] = {
    [TARGET_ARCH_XS3A] = {SATURATION_MIN_OFFSET_SYMMETRIC,
                          VLMACC_PRODUCT_LSB_DROPS_NONE,
                          VLMUL_SHR_S8_XS3A,
                          VLMUL_SHR_S16_XS3A,
                          VLMUL_SHR_S32},
    [TARGET_ARCH_VX4A] = {SATURATION_MIN_OFFSET_TWOS_COMPLEMENT,
                          VLMACC_PRODUCT_LSB_DROPS_ONE,
                          VLMUL_SHR_S8_VX4A,
                          VLMUL_SHR_S16_VX4A,
                          VLMUL_SHR_S32},
};

const nn_arch_profile_t *GetNNArchProfile(void) {
    return &NN_ARCH_PROFILES[NN_ARCH];
}

void SetNNTargetArch(nn_target_arch_t arch) {
    NN_ARCH = arch;
}
