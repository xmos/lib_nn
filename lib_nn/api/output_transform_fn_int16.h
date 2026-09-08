// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef _output_transform_fn_int16_h_
#define _output_transform_fn_int16_h_

#include <stdint.h>

typedef struct {
    int32_t output_slice_channel_count;
} otfn_int16_params_t;


/** Transform up to 16 ring-buffer accumulators into signed 16-bit outputs.
 *
 * Each accumulator is reconstructed from its low half in vR and its high half
 * in vD. An accumulator-domain bias is added with signed 32-bit saturation,
 * the result is multiplied by a Q2.30 multiplier with rounding and signed
 * 32-bit saturation, and the result is saturated to the signed 16-bit range
 * [-32768, 32767].
 *
 * The 32 elements in each vDvR group are ordered as:
 *
 * \code
 * vR0, vR1, ... vR15, vD0, vD1, ... vD15
 * \endcode
 *
 * where vR contains the low 16 bits and vD contains the high 16 bits. The
 * buffer must contain all 32 elements even when fewer than 16 outputs are
 * requested. Its contents are not preserved by all implementations.
 * The accumulator pointer must be eight-byte aligned and the output pointer
 * must be word aligned. The output buffer must provide at least 16 writable
 * int16_t elements from output, although only the requested elements are
 * modified.
 *
 * For each group of 16 output channels, mul_add contains four vectors of eight
 * signed 32-bit elements:
 *
 * \code
 * a1, a3, a5, a7, a9, a11, a13, a15,
 * m1, m3, m5, m7, m9, m11, m13, m15,
 * a0, a2, a4, a6, a8, a10, a12, a14,
 * m0, m2, m4, m6, m8, m10, m12, m14
 * \endcode
 *
 * where aK is an accumulator-domain bias and mK is a Q2.30 multiplier for
 * output channel K. Output values are written in normal channel order.
 *
 * \param params               Output slice parameters. The channel count
 *                             determines how many values are written.
 * \param output               Destination for the current output group, with
 *                             capacity for at least 16 elements.
 * \param vDvR                 The eight-byte-aligned, 32-element vR/vD
 *                             accumulator buffer.
 * \param output_channel_group Zero-based group of 16 output channels.
 * \param mul_add              Packed biases and multipliers for all groups.
 *
 * \returns A pointer immediately after the last output value written.
 */
extern int16_t *output_transform_fn_int16(otfn_int16_params_t *params,
                                          int16_t *output,
                                          int16_t *vDvR,
                                          int32_t output_channel_group,
                                          int32_t *mul_add);

#endif
