// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_OP_UTILS_H_
#define NN_OP_UTILS_H_

#include <stdint.h>
#include <string.h>

#include "nn_api.h"

/**
 * @brief Split a range between threads using four-element alignment.
 *
 * The range [0, split_size) is divided into contiguous, non-overlapping
 * subranges. The first element of each subrange is aligned to four elements;
 * the final subrange end is set to split_size and may be unaligned. The
 * function may use fewer than tc threads when split_size does not provide
 * enough aligned work for every requested thread.
 *
 * @param[in]  tc          Maximum number of threads to use
 * @param[in]  split_size  Number of elements to split
 * @param[out] split_start Start indices for each subrange; provide room for tc entries
 * @param[out] split_end   Exclusive end indices for each subrange; provide room for tc entries
 * @return The number of subranges written to split_start and split_end
 */
C_API int calculateAlignedThreadSplit(int tc, int split_size, int split_start[], int split_end[]);

/**
 * @brief Split a range between threads while aligning subrange starts.
 *
 * The range [0, split_size) is divided into contiguous, non-overlapping
 * subranges. Subrange starts are rounded to alignment boundaries, and the
 * final subrange end is split_size. alignment must be a positive power of two.
 * The function may use fewer than tc threads when split_size does not provide
 * enough aligned work for every requested thread.
 *
 * @param[in]  tc          Maximum number of threads to use
 * @param[in]  split_size  Number of elements to split
 * @param[out] split_start Start indices for each subrange; provide room for tc entries
 * @param[out] split_end   Exclusive end indices for each subrange; provide room for tc entries
 * @param[in]  alignment   Alignment, in elements; must be a positive power of two
 * @return The number of subranges written to split_start and split_end
 */
C_API int calculateThreadSplit(int tc, int split_size, int split_start[], int split_end[], int alignment);

#endif // NN_OP_UTILS_H_
