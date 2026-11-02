// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * @file nn_arch.h
 * @brief Runtime architecture profiles for VPU behavior.
 */
#pragma once

#include "nn_api.h"

/** @brief Architectures with runtime-selectable VPU behavior. */
typedef enum {
  TARGET_ARCH_XS3A = 0, ///< XS3A VPU behavior.
  TARGET_ARCH_VX4A = 1, ///< VX4A VPU behavior.
  TARGET_ARCH_COUNT,    ///< Number of supported architecture profiles.
} nn_target_arch_t;

/** @brief Offset from the negated positive maximum for signed saturation. */
typedef enum {
  SATURATION_MIN_OFFSET_SYMMETRIC = 0, ///< Use symmetric saturation.
  SATURATION_MIN_OFFSET_TWOS_COMPLEMENT = -1, ///< Include the most negative value.
} nn_saturation_min_offset_t;

/** @brief Number of least-significant product bits discarded by VLMACC. */
typedef enum {
  VLMACC_PRODUCT_LSB_DROPS_NONE = 0, ///< Preserve every product bit.
  VLMACC_PRODUCT_LSB_DROPS_ONE = 1,  ///< Discard one product bit.
} nn_vlmacc_product_lsb_drops_t;

/** @brief Right shifts applied to signed 8-bit VLMUL products. */
typedef enum {
  VLMUL_SHR_S8_XS3A = 6, ///< XS3A signed 8-bit VLMUL shift.
  VLMUL_SHR_S8_VX4A = 7, ///< VX4A signed 8-bit VLMUL shift.
} nn_vlmul_shr_s8_t;

/** @brief Right shifts applied to signed 16-bit VLMUL products. */
typedef enum {
  VLMUL_SHR_S16_XS3A = 14, ///< XS3A signed 16-bit VLMUL shift.
  VLMUL_SHR_S16_VX4A = 15, ///< VX4A signed 16-bit VLMUL shift.
} nn_vlmul_shr_s16_t;

/** @brief Temporary compatibility alias for nn_vlmul_shr_s16_t. */
#define VLMUL_SHR_XS3A VLMUL_SHR_S16_XS3A
#define VLMUL_SHR_VX4A VLMUL_SHR_S16_VX4A
typedef nn_vlmul_shr_s16_t nn_vlmul_shr_t;

/** @brief Right shifts applied to signed 32-bit VLMUL products. */
typedef enum {
  VLMUL_SHR_S32 = 30, ///< Signed 32-bit VLMUL shift for all profiles.
} nn_vlmul_shr_s32_t;

/** @brief VPU behavior associated with one runtime architecture. */
typedef struct {
  nn_saturation_min_offset_t saturation_min_offset; ///< Signed saturation minimum offset.
  nn_vlmacc_product_lsb_drops_t vlmacc_product_lsb_drops; ///< Discarded VLMACC product bits.
  nn_vlmul_shr_s8_t vlmul_shr_s8;   ///< Signed 8-bit VLMUL right shift.
  nn_vlmul_shr_s16_t vlmul_shr_s16; ///< Signed 16-bit VLMUL right shift.
  nn_vlmul_shr_s32_t vlmul_shr_s32; ///< Signed 32-bit VLMUL right shift.
} nn_arch_profile_t;

/**
 * @brief Currently selected runtime architecture.
 *
 * This process-wide value is read by VPU simulation functions. Concurrent
 * reads are safe, but changing it while an operation is using it causes a data
 * race and may mix architecture behavior within that operation.
 */
extern nn_target_arch_t NN_ARCH;

/**
 * @brief Return the VPU behavior profile for the selected architecture.
 *
 * @return Read-only profile selected by NN_ARCH.
 */
C_API const nn_arch_profile_t *GetNNArchProfile(void);

/**
 * @brief Select the runtime architecture used by VPU simulation.
 *
 * This function changes process-wide state and is not reentrant with
 * architecture-dependent operations. Call it only when no other thread or
 * operation can read NN_ARCH or its selected profile.
 *
 * @param arch Architecture profile to select.
 */
C_API void SetNNTargetArch(nn_target_arch_t arch);
