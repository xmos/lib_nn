// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_OP_UTILS_H_
#define NN_OP_UTILS_H_

#include <stdint.h>
#include <string.h>

#include "vpu_memcpy.h"
#include "xs3_vpu.h"

C_API int calculateAlignedThreadSplit(int tc, int split_size, int split_start[], int split_end[]);
C_API int calculateThreadSplit(int tc, int split_size, int split_start[], int split_end[], int alignment);

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
#define IMG_ADDRESS_VECT(IMG, DELTA_ROWS, DELTA_COLS, DELTA_CHANS)             \
  (((DELTA_ROWS) * (IMG)->width * (IMG)->channels) +                           \
   ((DELTA_COLS) * (IMG)->channels) + (DELTA_CHANS))

/** Get the number of output channel groups given the number of output channels.
 *
 * This macro gives the minimum number of groups required to handle `CHANNELS`
 * channels, which means it is effectively `(int) ceil(CHANNELS / 16.0f)`.
 *
 * \param CHANNELS  Number of channels
 */
#define OUT_CHANNEL_GROUPS(CHANNELS)                                           \
  (((CHANNELS) + (VPU_INT8_ACC_PERIOD - 1)) >> VPU_INT8_ACC_PERIOD_LOG2)

#ifdef NN_USE_REF
#define USING_C_REFERENCE (1)
#else
#define USING_C_REFERENCE (0)
#endif // NN_USE_REF

/**
 * @brief set `word_count` words from `value` to `dst`.
 *
 * `dst` must be a word-aligned address.
 *
 * @param dst  [out]    Destination address, must be word aligned
 * @param value  [in]   Source value.
 * @param size [in]     Number of 32 bit words to be copied
 */
void vpu_memset_32(void *dst, const int32_t value, const int word_count);

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
void vpu_memset_vector(void *dst, const int32_t value, const int vector_count);

#ifdef __XC__
} // extern "C"
#endif

#endif // NN_OP_UTILS_H_
