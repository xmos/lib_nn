// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef LAYERS_H_
#define LAYERS_H_
#include "nn_api.h"
#include "nn_bin_types.h"
#include "nn_image.h"
#include <string.h>

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
void bsign_8_prepare(nn_bsign_8_job_t *jobs, int8_t *zero_point_vect,
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
void bsign_8(bnn_b32_t *Y, const int8_t *X, const int8_t *zero_point_vect,
             const nn_bsign_8_job_t *job);

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
C_API void pad_prepare(nn_pad_plan_t *plan, const padding_sizes_t *p,
                       const nn_image_params_t *x,
                       const unsigned bytes_per_pixel);

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
void pad_run(char *y, char *x, const nn_pad_plan_t *p, uint32_t pad_value);

void pad_ref(char *y, char *x, const padding_sizes_t *p,
             const nn_image_params_t *xp, const unsigned bytes_per_pixel,
             uint32_t pad_value);

/**
 * Func to calculate n_3
 */
void pad_3_to_4_prepare(uint32_t *n_3, const unsigned height,
                        const unsigned width);

/** Function that pads an image with 3-byte values with a 0.
 * The output image must be word aligned. This function solves the general
 * case and calls an optimised assembly version for the bulk copy.
 *
 * @param    outputs    output values, every word contains 3 bytes and a zero
 * @param    inputs     input values, RGBRGBRGBRGB...
 * @param    N_3        number of blocks of 3 bytes to copy
 *
 * @returns  The inner product
 */
extern void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3,
                           uint32_t pad_val);
extern void pad_3_to_4_ref(int8_t outputs[], int8_t inputs[], uint32_t N_3,
                           uint32_t pad_val);

typedef struct nn_mul_params_t {
  int8_t in1_zero_point;
  int8_t in2_zero_point;
  int16_t bias;
  int16_t scalar;
  int16_t vlashr_shr;
} nn_mul_params_t;

void mul_boggle(nn_mul_params_t *params, double in1Scale, double in2Scale,
                double outputScale, int8_t in1ZeroPoint, int8_t in2ZeroPoint,
                int8_t outputZeroPoint);
void mul_elementwise(const int8_t *in1_data, const int8_t *in2_data,
                     int element_count, nn_mul_params_t *params,
                     int8_t *out_data);

// /**
//  * Describes the parameters needed for an @oper{add_elementwise} operator.
//  @see add_elementwise().
//  */
// typedef struct {
//     /**
//      * The parameters that are applied to each input element.
//      */
//
//     /**
//     * `m1` and `m2` are the multiplers for the inputs.
//     */
//     int16_t m1[16];
//     int16_t m2[16];

//     /**
//     * `shift` is the number of bits the 32-bit accumulator is
//     * right-shifted by to obtain a final result for each element.
//     */
//     int16_t shift[16];

//     /**
//     * `bias_hi` and `bias_lo` are together, the 32-bit bias to
//     * which the scaled inputs are added.
//     */
//     int16_t bias_lo[16];
//     int16_t bias_hi[16];

// } nn_add_params_t;

typedef struct {
  int16_t m1[16];
  int16_t m2[16];
  int16_t shift[16];
  int16_t bias_hi[16];
  int16_t bias_lo[16];
} nn_add_params_t;

/**
 * @brief Invoke an @oper{add_elementwise} job.
 *
 * The @oper{add_elementwise} operator adds together two quantized 8-bit input
 * vectors, @tensor{x_0} and @tensor{x_1} element-by-element to produce the
 * output vector @tensor{y}. This function assumes that the input vectors and
 * the output vector each require different quantization parameters.
 *
 * In order to add together two quantized vectors, their quantization parameters
 * must match. The contents of `params` indicate how to do this.
 *
 * @par Parameter Details
 *
 * `Y` points to the output vector @tensor{y} with shape @tensor_shape{N}.
 *
 * `X0` and `X1` respectively point to the first and second input vectors
 * @tensor{x_0} and @tensor{x_1}, each with shape
 * @tensor_shape{N}.
 *
 * `params` describes the parameters @math{s_i}, @math{m_i}, @math{b} and
 * @math{s_{out}} which are applied for each output element.
 *
 * `elm_start` and `elm_count` together specify which output elements
 * @math{y[k]} should be calculated by this invocation. Specifically, this
 * invocation will calculate @math{y[k]} for which `elm_start` @math{\le k \lt}
 * `(elm_start + elm_count)`.
 *
 * @param[out]  Y           The output vector @tensor{y}
 * @param[in]   X0          The first input vector @tensor{x_0}
 * @param[in]   X1          The second input vector @tensor{x_1}
 * @param[in]   params      The scaling and bias parameters
 * @param[in]   elm_start   Index of first output element to be computed
 * @param[in]   elm_count   Number of output elements to be computed
 */
void add_elementwise(int8_t Y[], const int8_t X1[], const int8_t X2[],
                     nn_add_params_t *p, const int elm_start,
                     const int elm_count);

/**
 * @brief Execute @oper{lookup8} job.
 *
 * See @oper_ref{lookup8} for more details about the @oper{lookup8} operator.
 *
 * Unlike other operators, instances of @oper{lookup8} do not require plans or
 * jobs and no initialization is necessary.
 *
 * `Y` points to the output vector @tensor{y} with length @math{N}.
 *
 * `X` points to the input vector @tensor{x} with length @math{N}.
 *
 * `lut` points to the look-up table @math{T} with shape @tensor_shape{256} and
 * dtype `int8`.
 *
 * `N` is the length @math{N} of the input vector @tensor{x}.
 *
 * @requires_word_alignment{Y,X}
 *
 * @param Y      [out]  The output vector @tensor{y}
 * @param X      [in]   The input vector @tensor{x}
 * @param lut    [in]   Look-up table @tensor{T}
 * @param N      [in]   Length @math{N} of input and output vectors
 */
void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
             const unsigned elm_start, const unsigned elm_count);

/**
 * @brief Execute @oper{softmax_exp_sum} job.
 *
 * `Y` points to the output scalar.
 *
 * `X` points to the input vector @tensor{x} with length @math{N}.
 *
 * `lut` points to the look-up table @math{T} with shape @tensor_shape{256} and
 * dtype `float32`.
 *
 * `N` is the length @math{N} of the input vector @tensor{x}.
 *
 * `elm_start` and `elm_count` together specify which output elements should be
 * summed into the output scalar.
 */
void softmax_exp_sum(float *Y, const int8_t *X, const float *lut,
                     const unsigned elm_start, const unsigned elm_count);

/**
 * @brief Execute @oper{softmax_exp_div} job.
 *
 * `Y` points to the output vector @tensor{y} with length @math{N}.
 *
 * `X` points to the input vector @tensor{x} with length @math{N}.
 *
 * `lut` points to the look-up table @math{T} with shape @tensor_shape{256} and
 * dtype `float32`.
 *
 * `inv_sum` is the reciprocal of the sum of the exponentials of the inputs.
 *
 * `elm_start` and `elm_count` together specify which output elements should be
 * calculated by this invocation.
 */
void softmax_exp_div(int8_t *Y, const int8_t *X, const float *lut,
                     const float inv_sum, const unsigned elm_start,
                     const unsigned elm_count);

void softmax_calculate_inv_sum(float *inv_sum, const float sums[]);

void softmax_generate_exp_lut(int zero_point, float scale, float *lut);

void softmax_ref(int8_t *Y, const int8_t *X, const float zero_point,
                 const float scale, const int length);

void slice_memcpy(int8_t *dst, int8_t *src, int32_t *in_offsets,
                  int32_t *out_offsets, int32_t *begin, int32_t *end,
                  void (*memcpy_func)(void *, void *, size_t));

void slice_memcpy_get_params(int *begin_dst, int *end_dst, int *in_offsets,
                             int *out_offsets, int *shape_dst, const int *begin,
                             const int *size, const int *shape,
                             const int dtype_size, const int rank);
#endif // LAYERS_H_
