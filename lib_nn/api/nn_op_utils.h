// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_OP_UTILS_H_
#define NN_OP_UTILS_H_

#include <stdint.h>
#include <string.h>

#include "nn_bso.h"
#include "xs3_vpu.h"

#ifdef __XC__
extern "C" {
#endif

/** Helper for computing offsets between pixels in an 8-bit image.
 *
 * Gives the address delta associated with moving the specified number of
 * rows, columns and channels through an image with the specified parameters.
 *
 * \param IMG           (nn_image_params_t*) Pointer to image params.
 * \param DELTA_ROWS    (signed int) Number of rows
 * \param DELTA_COLS    (signed int) Number of columns
 * \param DELTA_CHANS   (signed int) Number of channels
 */
#define IMG_ADDRESS_VECT(IMG, DELTA_ROWS, DELTA_COLS, DELTA_CHANS) \
  (((DELTA_ROWS) * (IMG)->width * (IMG)->channels) +               \
   ((DELTA_COLS) * (IMG)->channels) + (DELTA_CHANS))

/** Get the number of output channel groups given the number of output channels.
 *
 * This macro gives the minimum number of groups required to handle `CHANNELS`
 * channels, which means it is effectively `(int) ceil(CHANNELS / 16.0f)`.
 *
 * \param CHANNELS  Number of channels
 */
#define OUT_CHANNEL_GROUPS(CHANNELS) \
  (((CHANNELS) + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2)

#ifdef NN_USE_REF
#define USING_C_REFERENCE (1)
#else
#define USING_C_REFERENCE (0)
#endif  // NN_USE_REF

/**
 * Lays out the biases, scales and offsets into the format required for the
 * standard bias- scale-offset tensor used by many of the API functions.
 *
 * This is a helper function which consumes separate vectors for each component
 * of the BSO tensor and rearranges it into the required format. Calls to this
 * function can be avoided if necessary by storing the data in the layout
 * specified in @ref bso_layout. To understand how these values are used, see
 * @ref out_shift_scale.
 *
 * `bias` is a pointer to a 1-D vector (length `C_out`) of 32-bit biases.
 * `shift1`, `scale`, `offset_scale`, `offset` and `shift2` are each pointers to
 * 1-D vectors (length `C_out`) of 16-bit elements. In each case, the `i`th
 * element corresponds to output channel `i`.
 *
 * If `bso_out` and `bias` do not point to the same memory location, it will be
 * assumed that the `bso_out` tensor does not overlap the memory of the `bias`,
 * `shift1`, `scale`, `offset_scale`, `offset` and `shift2` vectors. In this
 * case, the `scratch` input will be ignored, and no memory will be allocated.
 *
 * If the `bso_out` and `bias` pointers do point to the same memory location, a
 * temporary scratch buffer is needed to perform the reformatting. In this case,
 * if the `scratch` parameter is not `NULL`, the memory to which it points will
 * be used as the scratch buffer. If the `scratch` parameter is `NULL`, memory
 * will be `malloc`ed for a scratch buffer. That memory will be `free`ed before
 * returning.
 *
 * The `bso_out` tensor must be large enough to hold `((C_out + 15)//16)`
 * elements of type `nn_bso_block_t`. `((C_out + 15)//16)*16` is the number of
 * output channels rounded up to the nearest multiple of
 * @ttref{VPU_INT8_ACC_PERIOD}.
 *
 * If `scratch` is provided, it must be large enough to store all `C_out`
 * elements of the `bias`, `shift1`, `scale`, `offset_scale`, `offset` and
 * `shift2` vectors.
 *
 * @param bso_out       The output tensor to be written
 * @param bias          The bias vector
 * @param shift1        The shift vector
 * @param scale         The scale vector
 * @param offset_scale  The offset scale
 * @param offset        The offset
 * @param shift2        The shift vector
 * @param scratch       An optional scratch buffer, or NULL
 * @param C_out         The number of output channels
 */
void nn_standard_BSO_layout(nn_bso_block_t* bso_out, int32_t* bias,
                            int16_t* shift1, int16_t* scale,
                            int16_t* offset_scale, int16_t* offset,
                            int16_t* shift2, data16_t* scratch,
                            const unsigned C_out);

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
void vpu_memcpy(void* dst, const void* src, size_t byte_count);

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
void vpu_memcpy_int(void* dst, const void* src, size_t byte_count);

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
void vpu_memcpy_ext(void* dst, const void* src, size_t byte_count);

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
void vpu_memcpy_vector_ext(void* dst, const void* src, int vector_count);

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
void vpu_memcpy_vector_int(void* dst, const void* src, int vector_count);

/**
 * @brief set `word_count` words from `value` to `dst`.
 *
 * `dst` must be a word-aligned address.
 *
 * @param dst  [out]    Destination address, must be word aligned
 * @param value  [in]   Source value.
 * @param size [in]     Number of 32 bit words to be copied
 */
void vpu_memset_32(void* dst, const int32_t value, const int word_count);

#define VPU_MEMSET_VECTOR_WORDS XS3_VPU_VREG_WIDTH_WORDS

/**
 * @brief set `vector_count` vector words from `value` to `dst`.
 *
 * `dst` must be a word-aligned address.
 *
 * @param dst  [out]            Destination address, must be word aligned
 * @param value  [in]           Source value.
 * @param vector_count [in]     Number of VPU_MEMSET_VECTOR_WORDS words vectors
 * to be copied.
 */
void vpu_memset_vector(void* dst, const int32_t value, const int vector_count);

#ifdef __XC__
}  // extern "C"
#endif

#endif  // NN_OP_UTILS_H_
