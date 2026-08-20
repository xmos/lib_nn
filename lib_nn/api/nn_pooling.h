// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef POOLING_H_
#define POOLING_H_

#include "nn_conv2d_structs.h"
#include "nn_image.h"
#include "nn_types.h"
#include "nn_window_params.h"

typedef nn_window_op_job_params_t nn_conv2d_job_params_t;

/**
 * Flags used with maxpool2d_ext() for advanced scenarios.
 */
typedef enum {
  /**
   * Placeholder flag used to indicate no other flags are needed.
   */
  MAXPOOL2D_FLAG_NONE = 0,
} nn_maxpool2d_flags_e;

/**
 * Flags used with avgpool2d_ext() for advanced scenarios.
 */
typedef enum {
  /**
   * Placeholder flag used to indicate no other flags are needed.
   */
  AVGPOOL2D_FLAG_NONE = 0,
} nn_avgpool2d_flags_e;

/**
 * Flags used with avgpool2d_global_ext() for advanced scenarios.
 */
typedef enum {
  /**
   * Placeholder flag used to indicate no other flags are needed.
   */
  AVGPOOL2D_GLOBAL_FLAG_NONE = 0,
} nn_avgpool2d_global_flags_e;

/**
 * @brief Perform 2D max pooling on an image.
 *
 * Slides a pooling window over the input image `X` and writes, for each output pixel, the maximum value found within the window to `Y`.
 *
 * `Y` points to the output image with shape (`y_params->height`, `y_params->width`, `x_params->channels`).
 *
 * `X` points to the input image with shape (`x_params->height`, `x_params->width`, `x_params->channels`). Both images use the standard image tensor memory layout (row-major, channels innermost).
 *
 * `pooling_window` describes the size and position of the pooling window and its stride relative to the input image:
 *   - `pooling_window->shape` is the height and width of the pooling window.
 *   - `pooling_window->start` is the row and column, in the input image's coordinate space, at which the pooling window starts for the top-left output pixel. For example, a `start` of `(0,0)` aligns the pooling window with the top-left corner of the input image with no implied padding, whereas `(1,1)` shifts it one pixel right and down.
 *   - `pooling_window->stride` is the vertical and horizontal number of pixels the pooling window moves for each output pixel.
 *   - `pooling_window->dilation` is ignored by this operator.
 *
 * The input and output images must have the same number of channels (`y_params->channels == x_params->channels`), and that channel count must be a multiple of 4 so that every pixel starts at a word-aligned address. `Y` and `X` must each point to a word-aligned address. Padding is not supported.
 *
 * Internally, `maxpool2d()` calls `maxpool2d_ext()` with a `job_params` argument that computes the entire output image, and with no flags set. To split the work into multiple invocations (e.g. for parallelization across cores), call `maxpool2d_ext()` directly.
 *
 * By default this operator saturates to the standard 8-bit limits ([-128, 127]). It can instead be configured to use symmetric saturation bounds ([-127, 127]) by defining `CONFIG_SYMMETRIC_SATURATION_maxpool2d` appropriately (see `nn_config.h`). This setting affects all instances of this operator.
 *
 * If the channel count is not a multiple of 32, this operator may read up to 28 bytes past the end of `X`. This is not ordinarily a problem, but if `X` is located very near the end of a valid memory range it is possible for a memory access exception to occur. If necessary, this can be avoided by reserving a buffer of up to 28 bytes immediately after `X`.
 *
 * @param[out]  Y               The output image
 * @param[in]   X               The input image
 * @param[in]   x_params        Parameters describing the shape of the input image
 * @param[in]   y_params        Parameters describing the shape of the output image
 * @param[in]   pooling_window  Parameters describing the relationship between the pooling window, the input image, and the output image
 */
void maxpool2d(nn_image_t* Y, const nn_image_t* X,
               const nn_image_params_t* x_params,
               const nn_image_params_t* y_params,
               const nn_window_params_t* pooling_window);

/**
 * @brief Perform a job (a subset of the output) of 2D max pooling on an image.
 *
 * This is the more flexible counterpart to maxpool2d(), allowing the output image to be computed by multiple invocations (e.g. one per core).
 *
 * `Y`, `X`, `x_params`, `y_params` and `pooling_window` are as described for maxpool2d().
 *
 * `job_params` indicates which output elements `Y[r,c,p]` this invocation computes:
 * @code
 *     job_params->start.rows <= r < job_params->start.rows + job_params->size.rows
 *     job_params->start.cols <= c < job_params->start.cols + job_params->size.cols
 *     job_params->start.channels <= p < job_params->start.channels + job_params->size.channels
 * @endcode
 *
 * `flags` is a collection of flags which modify the behavior of this operator; see `nn_maxpool2d_flags_e`. `MAXPOOL2D_FLAG_NONE` (0) gives the default behavior.
 *
 * The parameter constraints and additional remarks described for maxpool2d() also apply here.
 *
 * @param[out]  Y               The output image
 * @param[in]   X               The input image
 * @param[in]   x_params        Parameters describing the shape of the input image
 * @param[in]   y_params        Parameters describing the shape of the output image
 * @param[in]   pooling_window  Parameters describing the relationship between the pooling window, the input image, and the output image
 * @param[in]   job_params      Indicates which output elements are computed by this invocation
 * @param[in]   flags           Flags which modify the behavior of this call
 */
void maxpool2d_ext(nn_image_t* Y, const nn_image_t* X,
                   const nn_image_params_t* x_params,
                   const nn_image_params_t* y_params,
                   const nn_window_params_t* window_config,
                   const nn_window_op_job_params_t* job_params,
                   const nn_maxpool2d_flags_e flags);

/**
 * @brief Compute a scaled, biased sum over the entire input image, per channel.
 *
 * For each channel, `avgpool2d_global()` sums the pixel values across the whole image, scales and biases the sum, and writes an 8-bit result to `Y`. This is typically used to implement global average pooling.
 *
 * `Y` points to the 8-bit output vector, with one element per channel (length `x_params->channels`).
 *
 * `X` points to the 8-bit input image with shape (`x_params->height`, `x_params->width`, `x_params->channels`), using the standard image tensor memory layout.
 *
 * `bias` is the 32-bit value each channel's accumulator is initialized with, before summing. Because the final right-shift by `shift` bits is applied after accumulation, an absolute output offset of `b0` requires a `bias` value of `b0 << shift`.
 *
 * `scale` is an 8-bit coefficient by which every input pixel value is multiplied before being added to the accumulator.
 *
 * `shift` is the (rounding, saturating) right-shift applied to each 32-bit accumulator to produce the final 8-bit result.
 *
 * `Y` and `X` must each point to a word-aligned address, and the channel count must be a multiple of 4 so that every pixel starts at a word-aligned address. Padding is not supported.
 *
 * Internally, `avgpool2d_global()` calls `avgpool2d_global_ext()` with `chan_start` of 0 and `chan_count` equal to the full channel count. To split the work across multiple invocations (e.g. for parallelization), call `avgpool2d_global_ext()` directly.
 *
 * By default this operator saturates to the standard 8-bit limits ([-128, 127]). It can instead be configured to use symmetric saturation bounds ([-127, 127]) by defining `CONFIG_SYMMETRIC_SATURATION_avgpool2d_global` appropriately (see `nn_config.h`). This setting affects all instances of this operator.
 *
 * If the channel count is not a multiple of 16, this operator may read up to 12 bytes past the end of `X`. This is not ordinarily a problem, but if `X` is located very near the end of a valid memory range it is possible for a memory access exception to occur. If necessary, this can be avoided by reserving a buffer of up to 12 bytes immediately after `X`.
 *
 * @param[out]  Y           The output vector
 * @param[in]   X           The input image
 * @param[in]   bias        Initial 32-bit accumulator value, shared by all channels
 * @param[in]   scale       The factor by which input pixel values are scaled
 * @param[in]   shift       The right-shift applied to the 32-bit accumulators to yield an 8-bit result
 * @param[in]   x_params    Parameters describing the shape of the input image
 */
void avgpool2d_global(nn_image_t* Y, const nn_image_t* X, const int32_t bias,
                      const int8_t scale, const uint16_t shift,
                      const nn_image_params_t* x_params);

/**
 * @brief Compute a job (a subset of the channels) of a scaled, biased sum over the entire input image.
 *
 * This is the more flexible counterpart to avgpool2d_global(), allowing the output vector to be computed by multiple invocations (e.g. one per core).
 *
 * `Y`, `X`, `bias`, `scale`, `shift` and `x_params` are as described for avgpool2d_global(). `Y` and `X` should each point to the start of their respective objects, even when the job being invoked does not start at channel 0.
 *
 * `chan_start` is the index of the first output channel computed by this invocation.
 *
 * `chan_count` is the number of channels computed by this invocation.
 *
 * `flags` is a collection of flags which modify the behavior of this operator; see `nn_avgpool2d_global_flags_e`. `AVGPOOL2D_GLOBAL_FLAG_NONE` (0) gives the default behavior.
 *
 * The parameter constraints and additional remarks described for avgpool2d_global() also apply here.
 *
 * @param[out]  Y           The output vector
 * @param[in]   X           The input image
 * @param[in]   bias        Initial 32-bit accumulator value, shared by all channels
 * @param[in]   scale       The factor by which input pixel values are scaled
 * @param[in]   shift       The right-shift applied to the 32-bit accumulators to yield an 8-bit result
 * @param[in]   x_params    Parameters describing the shape of the input image
 * @param[in]   chan_start  Index of the first output channel to be computed
 * @param[in]   chan_count  Number of output channels to be computed
 * @param[in]   flags       Flags which modify the behavior of this call
 */
void avgpool2d_global_ext(nn_image_t* Y, const nn_image_t* X,
                          const int32_t bias, const int8_t scale,
                          const uint16_t shift,
                          const nn_image_params_t* x_params,
                          const unsigned chan_start, const unsigned chan_count,
                          const nn_avgpool2d_global_flags_e flags);

#endif  // POOLING_H_
