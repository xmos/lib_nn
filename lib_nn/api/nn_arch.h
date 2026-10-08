// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include "nn_api.h"

typedef enum {
  TARGET_ARCH_XS3A = 0,
  TARGET_ARCH_VX4A = 1,
} nn_target_arch_t;

extern nn_target_arch_t NN_ARCH;

C_API void SetNNTargetArch(nn_target_arch_t arch);

typedef struct {
  unsigned symmetric_saturation;
  unsigned vlmacc_product_lsb_drops;
  unsigned vlmul_shift_offset;
} nn_vpu_config_t;

extern nn_vpu_config_t NN_VPU_CONFIG;

C_API void SetNNVPUConfig(nn_vpu_config_t config);

typedef enum {
  VLMUL_SHR_XS3A = 14,
  VLMUL_SHR_VX4A = 15,
} nn_vlmul_shr_t;