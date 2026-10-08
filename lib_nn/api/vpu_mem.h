// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "nn_api.h"
#include "xs3_vpu.h"

/** @name Defines
 * @{ */

#if defined(__XS3A__) || defined(__riscv_xxcore)
#define MEMCPY_FPTRGROUP __attribute__((fptrgroup("memcpy_fn_group")))
#else
#define MEMCPY_FPTRGROUP
#endif

typedef void (*memcpy_fn_t)(void *dst, const void *src, size_t byte_count);

#define MEMCPY_VECT_EXT_BYTES (128)
#define MEMCPY_VECT_INT_BYTES (32)
#define VPU_MEMSET_VECTOR_WORDS XS3_VPU_VREG_WIDTH_WORDS

/** Replicate a byte across a 32-bit word. */
#define BROADCAST_8_TO_32(f) (((uint8_t)f) * 0x01010101)

/** Replicate a 16-bit value across a 32-bit word. */
#define BROADCAST_16_TO_32(f) (((uint16_t)f) * 0x00010001)

/** Replicate a byte across a 16-bit value. */
#define BROADCAST_8_TO_16(f) (((uint8_t)f) * 0x00000101)

/** @} */

/** @name Copy
 * @{ */

/**
 * @brief Copy bytes from internal SRAM.
 *
 * `dst` and `src` must be word aligned. The byte count need not be a
 * multiple of four.
 *
 * @param dst        [out] Destination address.
 * @param src        [in] Source address.
 * @param byte_count [in] Number of bytes to copy.
 */
C_API MEMCPY_FPTRGROUP
void vpu_memcpy_int(void *dst, const void *src, size_t byte_count);

/**
 * @brief Copy bytes from external flash or DDR.
 *
 * `dst` and `src` must be word aligned. The byte count need not be a
 * multiple of four.
 *
 * @param dst        [out] Destination address.
 * @param src        [in] Source address.
 * @param byte_count [in] Number of bytes to copy.
 */
C_API MEMCPY_FPTRGROUP
void vpu_memcpy_ext(void *dst, const void *src, size_t byte_count);

/**
 * @brief Copy blocks of MEMCPY_VECT_EXT_BYTES bytes from external flash or DDR.
 *
 * `dst` and `src` must be word aligned.
 *
 * @param dst          [out] Destination address.
 * @param src          [in] Source address.
 * @param vector_count [in] Number of MEMCPY_VECT_EXT_BYTES-byte blocks to copy.
 */
C_API MEMCPY_FPTRGROUP
void vpu_memcpy_vector_ext(void *dst, const void *src, size_t vector_count);

/**
 * @brief Copy blocks of MEMCPY_VECT_INT_BYTES bytes from internal SRAM.
 *
 * `dst` and `src` must be word aligned.
 *
 * @param dst          [out] Destination address.
 * @param src          [in] Source address.
 * @param vector_count [in] Number of MEMCPY_VECT_INT_BYTES-byte blocks to copy.
 */
C_API MEMCPY_FPTRGROUP
void vpu_memcpy_vector_int(void *dst, const void *src, size_t vector_count);

/** @} */

/** @name Move
 * @{ */

/**
 * @brief Move bytes between possibly overlapping regions.
 *
 * `dst` and `src` must be word aligned. Any number of bytes may be moved.
 *
 * @param dst        [out] Destination address.
 * @param src        [in] Source address.
 * @param byte_count [in] Number of bytes to move; may be zero.
 */
C_API void vpu_memmove_word_aligned(void *dst, const void *src,
                                    unsigned int byte_count);

/** @} */

/** @name Fill
 * @{ */

/**
 * @brief Fill 32-bit words with a repeated value.
 *
 * `dst` must be word aligned.
 *
 * @param dst        [out] Destination address.
 * @param value      [in] Value to repeat.
 * @param word_count [in] Number of 32-bit words to write.
 */
C_API void vpu_memset_32(void *dst, const int32_t value, const int word_count);

/**
 * @brief Fill vectors with a repeated 32-bit value.
 *
 * `dst` must be word aligned.
 *
 * @param dst          [out] Destination address.
 * @param value        [in] Value to repeat.
 * @param vector_count [in] Number of VPU_MEMSET_VECTOR_WORDS-word vectors.
 */
C_API void vpu_memset_vector(void *dst, const int32_t value,
                            const int vector_count);

/**
 * @brief Fill bytes from a repeated 32-byte pattern buffer.
 *
 * `src` must be word aligned. The destination is assumed to follow the
 * source vector's repeated-byte pattern, with its starting position selected
 * according to the destination's offset within a word.
 *
 * @param dst        [out] Destination address.
 * @param src        [in] Address of the 32-byte pattern buffer.
 * @param byte_count [in] Number of bytes to fill.
 */
C_API void vpu_memset_256(void *dst, const void *src,
                         unsigned int byte_count);

/**
 * @brief Repeat a 32-bit value across a 32-byte buffer.
 *
 * BROADCAST_8_TO_32() and BROADCAST_16_TO_32() prepare values for byte and
 * 16-bit fills through vpu_memset_256().
 *
 * @param dst  [out] Word-aligned address of a 32-byte buffer.
 * @param from [in] Value to repeat.
 */
C_API void broadcast_32_to_256(void *dst, uint32_t from);

/** @} */
