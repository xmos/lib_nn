#pragma once

#include "nn_api.h"

typedef enum {
  TARGET_ARCH_XS3A = 0,
  TARGET_ARCH_VX4A = 1,
} nn_target_arch_t;

extern nn_target_arch_t NN_ARCH;

C_API void SetNNTargetArch(nn_target_arch_t arch);

typedef enum {
  VLMUL_SHR_XS3A = 14,
  VLMUL_SHR_VX4A = 15,
} nn_vlmul_shr_t;