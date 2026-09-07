// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <stdint.h>

#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif

/**
 * @brief set `word_count` words from `value` to `dst`.
 *
 * `dst` must be a word-aligned address.
 *
 * @param dst        [out] Destination address, must be word aligned.
 * @param value      [in]  Source value.
 * @param word_count [in]  Number of 32-bit words to be written.
 */
void vpu_memset_32(void *dst, const int32_t value, const int word_count);

#define VPU_MEMSET_VECTOR_WORDS XS3_VPU_VREG_WIDTH_WORDS

/**
 * @brief set `vector_count` vectors from `value` to `dst`.
 *
 * `dst` must be a word-aligned address.
 *
 * @param dst          [out] Destination address, must be word aligned.
 * @param value        [in]  Source value.
 * @param vector_count [in]  Number of VPU_MEMSET_VECTOR_WORDS-word vectors to
 *                          be written.
 */
void vpu_memset_vector(void *dst, const int32_t value, const int vector_count);

/**
 * Fill `byte_count` bytes by repeating the byte selected from `src` according to
 * the VPU vector replication pattern.
 *
 * `src` must be word aligned. The destination is assumed to be laid out in the
 * same repeated-byte pattern as the source vector.
 */
void vpu_memset_256(void *dst, const void *src, unsigned int byte_count);

/**
 * Broadcast a 32-bit value across an 256-bit vector.
 */
void broadcast_32_to_256(void *dst, uint32_t from);

/**
 * Macro that replicates a byte over an int.
 * Use with broadcast_32_to_256() in order to replicate a byte over a vector.
 */
#define BROADCAST_8_TO_32(f) (((uint8_t)f) * 0x01010101)

/**
 * Macro that replicates a short over an int.
 * Use with broadcast_32_to_256() in order to replicate a short over a vector.
 */
#define BROADCAST_16_TO_32(f) (((uint16_t)f) * 0x00010001)

/**
 * Macro that replicates a byte over a short.
 */
#define BROADCAST_8_TO_16(f) (((uint8_t)f) * 0x00000101)

#ifdef __XC__
} // extern "C"
#endif
