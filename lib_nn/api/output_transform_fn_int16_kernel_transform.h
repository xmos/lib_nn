// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef _output_transform_fn_int16_kernel_transform_h_
#define _output_transform_fn_int16_kernel_transform_h_

#include <stdint.h>

/** Prepare per-channel parameters for the signed 16-bit output transform.
 *
 * Floating-point channel multipliers are converted to Q2.30. For every group
 * of 16 channels, biases and multipliers are packed as:
 *
 * \code
 * a1, a3, ... a15, m1, m3, ... m15,
 * a0, a2, ... a14, m0, m2, ... m14
 * \endcode
 *
 * Biases are signed accumulator-domain values that output_transform_fn_int16()
 * adds before applying the corresponding multiplier.
 *
 * kernel_weights_in, kernel_weights_out, and input_channels are retained for
 * API compatibility but are not used.
 *
 * @param kernel_weights_in      Unused.
 * @param channel_multipliers_in Per-channel floating-point multipliers.
 * @param channel_bias_terms_in  Per-channel accumulator-domain biases.
 * @param kernel_weights_out     Unused.
 * @param mul_add_out            Output buffer of 32 elements per group of up
 *                               to 16 output channels.
 * @param input_channels         Unused.
 * @param output_channels        Number of output channels to prepare.
 */
void output_transform_fn_int16_kernel_transform(
    const int8_t *kernel_weights_in,
    const float *channel_multipliers_in, const int *channel_bias_terms_in,
    int8_t *kernel_weights_out, int32_t *mul_add_out,
    int input_channels, int output_channels);

#endif
