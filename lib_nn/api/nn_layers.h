// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef LAYERS_H_
#define LAYERS_H_
#include "nn_image.h"
#include "nn_bin_types.h"

/**
 * Struct represents the parameters needed by each `bsign_8()` job.
 *
 * Values are set by `bsign_8_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  mem_stride_t start;
  int32_t length;
} nn_bsign_8_job_t;

/**
 * @brief Initialize an instance of the @oper{bsign_8} operator.
 *
 * See @oper_ref{bsign_8} for more details about the @oper{bsign_8} operator. To
 * invoke a
 * @oper{bsign_8} job, call bsign_8().
 *
 * When bsign_8() is called, a job (`nn_bsign_8_job_t`) must be supplied to tell
 * it how to do its work. This function initializes one or more jobs to be
 * supplied in subsequent calls to bsign_8().
 *
 * Each job computes a range of elements in the output vector (possibly the
 * entire vector).
 *
 * `jobs` points to an array of `nn_bsign_8_t` to be initialized. Each element
 * represents one job. There should be `job_count` elements in the array.
 *
 * `N` is the number of elements @math{N} in the input vector @tensor{x} and
 * output vector @tensor{y}.
 *
 * `job_count` indicates the number of jobs to be initialized (and thus the
 * number of elements in the `jobs` array).
 *
 * Unlike many other operators, @oper{bsign_8} will automatically divide the
 * work to be done as evenly as possible between jobs.
 *
 * @param plan      [out]  The plan to be initialized.
 * @param jobs      [out]   Array of jobs to be initialized.
 * @param N         [in]    The number of elements in the input.
 * @param[in]  zero_point   The value @math{z_0} to be used for padding (for all
 * channels)
 * @param job_count [in]    The number of jobs to be initialized.
 */
void bsign_8_prepare(nn_bsign_8_job_t* jobs, int8_t* zero_point_vect,
                     const uint32_t N, const int8_t zero_point,
                     const int32_t job_count);

/**
 * @brief Execute @oper{bsign_8} job.
 *
 * See @oper_ref{bsign_8} for more details about the @oper{requantize_16_to_8}
 * operator.
 *
 * An instance of the @oper{bsign_8} operator requires an job (but no plan is
 * required). See bsign_8_prepare() for more details.
 *
 * `Y` points to the output vector @tensor{y} with length @math{N}. The address
 * supplied for `Y` should be the start address of the output vector (for any
 * job being processed).
 *
 * `X` points to the input vector @tensor{x} with length @math{N}. The address
 * supplied for `X` should be the start address of the input vector (for any job
 * being processed).
 *
 * `job` points to the (initialized) @oper{bsign_8} job to be performed with
 * this call.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y   [out]    The output vector @tensor{y}
 * @param X   [in]     The input vector @tensor{x}
 * @param plan [in]    The @oper{bsign_8} plan to be processed
 * @param job [in]     The @oper{bsign_8} job to be processed
 */
void bsign_8(bnn_b32_t* Y, const int8_t* X, const int8_t* zero_point_vect,
             const nn_bsign_8_job_t* job);

/**
 * Struct represents the parameters needed by each `pad_run()` job.
 *
 * Values are set by `pad_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct nn_pad_plan_t {
  unsigned top_pad_bytes;
  unsigned mid_loop_count;
  unsigned left_pad_bytes;
  unsigned mid_copy_bytes;
  unsigned right_pad_bytes;
  unsigned bottom_pad_bytes;
} nn_pad_plan_t;

typedef struct padding_sizes_t {
  int32_t top;
  int32_t bottom;
  int32_t left;
  int32_t right;
} padding_sizes_t;

/**
 * @brief Execute @oper{pad_prepare} function.
 *
 * `plan` points to the output vector @tensor{y} with length @math{N}.
 *
 * `p` struct describing the padding to be applied to the input tensor.
 *
 * `x` parameters describing the input tensor to be padded.
 *
 * `bytes_per_pixel` the bytes per pixel for tensor x.
 *
 * @param plan             [out]  The output vector @tensor{y}
 * @param p                [in]   The input vector @tensor{x}
 * @param x                [in]   Look-up table @tensor{T}
 * @param bytes_per_pixel  [in]   Length @math{N} of input and output vectors
 */
void pad_prepare(nn_pad_plan_t* plan, const padding_sizes_t* p,
                 const nn_image_params_t* x, const unsigned bytes_per_pixel);

/**
 * @brief Execute @oper{pad_run} job.
 *
 * See @oper_ref{pad_run} for more details about the @oper{requantize_16_to_8}
 * operator.
 *
 * `Y` points to the output vector @tensor{y}.
 *
 * `X` points to the input vector @tensor{x}.
 *
 * `plan` points to the (initialized) plan.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param y   [out]    The output vector @tensor{y}
 * @param x   [in]     The input vector @tensor{x}
 * @param plan [in]    The prameters describing how to pad.
 */
void pad_run(char* y, char* x, const nn_pad_plan_t* p, uint32_t pad_value);

void pad_ref(char* y, char* x, const padding_sizes_t* p,
             const nn_image_params_t* xp, const unsigned bytes_per_pixel,
             uint32_t pad_value);

#endif  // LAYERS_H_
