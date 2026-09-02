// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef VPU_MEMCPY_H_
#define VPU_MEMCPY_H_

#include <stddef.h>

#ifdef __XC__
extern "C" {
#endif

// fptrgroup is an xcore-only extension; guard it so native builds still see
// a valid (empty) attribute.
#if (defined(__XS3A__) || defined(__VX4B__))
#define MEMCPY_FPTRGROUP __attribute__((fptrgroup("memcpy_fn_group")))
#else
#define MEMCPY_FPTRGROUP
#endif

typedef void (*memcpy_fn_t)(void *dst, const void *src, size_t byte_count);

#define MEMCPY_VECT_EXT_BYTES (128)
#define MEMCPY_VECT_INT_BYTES (32)

/**
 * @brief Copy `size` bytes from `src` to `dst`.
 *
 * `dst` and `src` both must be word-aligned addresses.
 *
 * `size` need not be an integer number of words.
 *
 * @param dst  [out]    Destination address
 * @param src  [in]     Source address
 * @param byte_count [in]     Number of bytes to be copied
 */
MEMCPY_FPTRGROUP
void vpu_memcpy(void *dst, const void *src, size_t byte_count);

/**
 * @brief Copy `size` bytes from `src` to `dst`.
 * Faster for copies from internal SRAM.
 *
 * `dst` and `src` both must be word-aligned addresses.
 *
 * `size` need not be an integer number of words.
 *
 * @param dst  [out]    Destination address
 * @param src  [in]     Source address
 * @param byte_count [in]     Number of bytes to be copied
 */
MEMCPY_FPTRGROUP
void vpu_memcpy_int(void *dst, const void *src, size_t byte_count);

/**
 * @brief Copy `size` bytes from `src` to `dst`.
 * Faster for copies from external flash and DDR.
 *
 * `dst` and `src` both must be word-aligned addresses.
 *
 * `size` need not be an integer number of words.
 *
 * @param dst  [out]    Destination address
 * @param src  [in]     Source address
 * @param byte_count [in]     Number of bytes to be copied
 */
MEMCPY_FPTRGROUP
void vpu_memcpy_ext(void *dst, const void *src, size_t byte_count);

/**
 * @brief Copy `vector_count` multiples of MEMCPY_VECT_EXT_BYTES bytes
 * from `src` to `dst`.
 * Faster for copies from external flash and DDR.
 *
 * `dst` and `src` both must be word-aligned addresses.
 *
 * `size` need not be an integer number of words.
 *
 * @param dst  [out]    Destination address
 * @param src  [in]     Source address
 * @param vector_count [in]     Number of MEMCPY_VECT_EXT_BYTES bytes copies to
 * be bytes to be performed
 */
MEMCPY_FPTRGROUP
void vpu_memcpy_vector_ext(void *dst, const void *src, size_t vector_count);

/**
 * @brief Copy `vector_count` multiples of MEMCPY_VECT_INT_BYTES bytes
 * from `src` to `dst`.
 * Faster for copies from internal SRAM.
 *
 * `dst` and `src` both must be word-aligned addresses.
 *
 * `size` need not be an integer number of words.
 *
 * @param dst  [out]    Destination address
 * @param src  [in]     Source address
 * @param vector_count [in]     Number of MEMCPY_VECT_INT_BYTES bytes copies to
 * be bytes to be performed
 */
MEMCPY_FPTRGROUP
void vpu_memcpy_vector_int(void *dst, const void *src, size_t vector_count);

#ifdef __XC__
} // extern "C"
#endif

#endif // VPU_MEMCPY_H_
