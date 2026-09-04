// Copyright 2020-2026 XMOS LIMITED.
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
 * @brief Initialize one or more jobs for bsign_8().
 *
 * `jobs` points to an array of `job_count` jobs to be initialized; each job computes a range of the output vector, and together they automatically divide the work as evenly as possible.
 *
 * `N` is the number of scalar elements in the input vector `X`; the bit-packed output `Y` requires `ceil(N / 32)` `bnn_b32_t` elements.
 *
 * `zero_point` is the value used for padding (for all channels).
 *
 * @param jobs        [out]  Array of jobs to be initialized
 * @param zero_point_vect [out] Padding value vector derived from `zero_point`
 * @param N           [in]   The number of elements in the input
 * @param zero_point  [in]   The value used for padding
 * @param job_count   [in]   The number of jobs to be initialized
 */
void bsign_8_prepare(nn_bsign_8_job_t *jobs, int8_t *zero_point_vect,
                     const uint32_t N, const int8_t zero_point,
                     const int32_t job_count);

/**
 * @brief Compute the bit-packed sign of each element of a vector.
 *
 * For each input element, writes a `1` bit to `Y` if the (zero-point adjusted) value is negative, else a `0` bit. No plan is required; see bsign_8_prepare() for job initialization.
 *
 * `Y` and `X` must each point to the start of their respective vectors (regardless of which job is being processed), and must each be word-aligned.
 *
 * @param Y               [out]  The output bit-packed vector
 * @param X               [in]   The input vector
 * @param zero_point_vect [in]   Per-channel zero-point vector from bsign_8_prepare()
 * @param job             [in]   The job to be processed
 */
void bsign_8(bnn_b32_t *Y, const int8_t *X, const int8_t *zero_point_vect,
             const nn_bsign_8_job_t *job);

/**
 * @brief Compute the number of 3-byte blocks to be copied by pad_3_to_4_run(), given an image's height and width.
 *
 * @param[out]  n_3     Number of 3-byte blocks
 * @param[in]   height  Image height, in pixels
 * @param[in]   width   Image width, in pixels
 */
void pad_3_to_4_prepare(uint32_t *n_3, const unsigned height,
                        const unsigned width);

/**
 * @brief Pad an image of 3-byte pixels out to 4 bytes per pixel, setting the added byte to a specified value.
 *
 * The output image must be word-aligned. This function handles the general case and calls an optimized assembly routine for the bulk copy.
 *
 * @param outputs  [out]  Output values; every word contains 3 bytes and a zero
 * @param inputs   [in]   Input values, e.g. RGBRGBRGBRGB...
 * @param N_3      [in]   Number of 3-byte blocks to copy
 * @param pad_val  [in]   Value written to the padding byte
 */
void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3,
                           uint32_t pad_val);

/**
 * @brief Pad a vector of bytes into 32-bit words, writing each input byte into the least-significant byte of an output word and filling the upper three bytes with the fixed padding value.
 *
 * The function processes `N * 4` input bytes and expands each byte into a 32-bit output word. `N` therefore counts 4-byte input chunks, not bytes.
 *
 * @param outputs  [out]  Output values; each word contains one input byte and three pad bytes
 * @param inputs   [in]   Input values
 * @param N        [in]   Number of 4-byte chunks to copy
 * @param pad_val  [in]   Value written to the upper three bytes of each output word
 */
void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N,
                          uint32_t pad_val);

typedef struct nn_mul_params_t {
  int8_t in1_zero_point;
  int8_t in2_zero_point;
  int16_t bias;
  int16_t scalar;
  int16_t vlashr_shr;
} nn_mul_params_t;

/**
 * @brief Compute the quantization parameters for mul_elementwise() from the inputs' and output's zero-points and scales.
 *
 * @param[out]  params        The computed parameters
 * @param[in]   in1Scale      Quantization scale of the first input
 * @param[in]   in2Scale      Quantization scale of the second input
 * @param[in]   outputScale   Quantization scale of the output
 * @param[in]   in1ZeroPoint  Quantization zero-point of the first input
 * @param[in]   in2ZeroPoint  Quantization zero-point of the second input
 * @param[in]   outputZeroPoint  Quantization zero-point of the output
 */
void mul_boggle(nn_mul_params_t *params, double in1Scale, double in2Scale,
                double outputScale, int8_t in1ZeroPoint, int8_t in2ZeroPoint,
                int8_t outputZeroPoint);

/**
 * @brief Multiply two quantized 8-bit input vectors element-by-element to produce a quantized 8-bit output vector.
 *
 * `params` (from mul_boggle()) describes how to reconcile the input and output quantization parameters.
 *
 * @param[in]   in1_data       The first input vector
 * @param[in]   in2_data       The second input vector
 * @param[in]   element_count  Number of elements to compute
 * @param[in]   params         The quantization parameters
 * @param[out]  out_data       The output vector
 */
void mul_elementwise(const int8_t *in1_data, const int8_t *in2_data,
                     int element_count, nn_mul_params_t *params,
                     int8_t *out_data);

typedef struct {
  int16_t m1[16];
  int16_t m2[16];
  int16_t shift[16];
  int16_t bias_hi[16];
  int16_t bias_lo[16];
} nn_add_params_t;

/**
 * @brief Add together two quantized 8-bit input vectors, element-by-element, to produce a quantized 8-bit output vector.
 *
 * This assumes the two input vectors and the output vector each require different quantization parameters; `params` describes how to reconcile them.
 *
 * `elm_start` and `elm_count` together specify which output elements `Y[k]` are computed by this invocation, namely those for which `elm_start <= k < elm_start + elm_count`.
 *
 * @param[out]  Y           The output vector
 * @param[in]   X1          The first input vector
 * @param[in]   X2          The second input vector
 * @param[in]   p           The scaling and bias parameters
 * @param[in]   elm_start   Index of first output element to be computed
 * @param[in]   elm_count   Number of output elements to be computed
 */
void add_elementwise(int8_t Y[], const int8_t X1[], const int8_t X2[],
                     nn_add_params_t *p, const int elm_start,
                     const int elm_count);

/**
 * @brief Apply an 8-bit look-up table to a vector, element-by-element.
 *
 * No plan or job initialization is required for this operator.
 *
 * `elm_start` and `elm_count` together specify which output elements `Y[k]` are computed by this invocation, namely those for which `elm_start <= k < elm_start + elm_count`. `Y` and `X` must each point to the start of their respective vectors, and must each be word-aligned.
 *
 * @param Y      [out]  The output vector
 * @param X      [in]   The input vector
 * @param lut    [in]   Look-up table with 256 `int8` entries
 * @param elm_start [in] Index of first output element to be computed
 * @param elm_count [in] Number of output elements to be computed
 */
void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
             const unsigned elm_start, const unsigned elm_count);

/**
 * @brief Sum the exponentials of a range of elements of a softmax input vector.
 *
 * `lut` is a 256-entry `float32` look-up table mapping each possible 8-bit input value to its exponential (see softmax_generate_exp_lut()). `elm_start` and `elm_count` together specify which input elements are summed into the output scalar.
 *
 * @param Y   [out]  The output scalar (sum of exponentials)
 * @param X   [in]   The input vector
 * @param lut [in]   Look-up table of exponentials
 * @param elm_start [in] Index of first input element to be summed
 * @param elm_count [in] Number of input elements to be summed
 */
void softmax_exp_sum(float *Y, const int8_t *X, const float *lut,
                     const unsigned elm_start, const unsigned elm_count);

/**
 * @brief Divide the exponential of each element of a softmax input vector by the sum of all exponentials, producing the final softmax output.
 *
 * `lut` is a 256-entry `float32` look-up table of exponentials (see softmax_generate_exp_lut()). `inv_sum` is 256 divided by the sum of the exponentials of the whole input vector (see softmax_calculate_inv_sum()). `elm_start` and `elm_count` together specify which output elements are computed by this invocation.
 *
 * @param Y       [out]  The output vector
 * @param X       [in]   The input vector
 * @param lut     [in]   Look-up table of exponentials
 * @param inv_sum [in]   Reciprocal of the sum of the exponentials of the inputs
 * @param elm_start [in] Index of first output element to be computed
 * @param elm_count [in] Number of output elements to be computed
 */
void softmax_exp_div(int8_t *Y, const int8_t *X, const float *lut,
                     const float inv_sum, const unsigned elm_start,
                     const unsigned elm_count);

/**
 * @brief Compute the reciprocal of the sum of a set of partial sums.
 *
 * Used to combine the per-job outputs of softmax_exp_sum() into a single scaling
 * factor for use by softmax_exp_div().
 *
 * `sums` must point to exactly 5 `float32` values (`sums[0]` through `sums[4]`).
 * The implementation reads those 5 entries unconditionally and ignores any
 * additional elements. The output is computed as
 * `inv_sum = 256.0f / (sums[0] + sums[1] + sums[2] + sums[3] + sums[4])`.
 *
 * @param[out]  inv_sum  The reciprocal of the total sum
 * @param[in]   sums     Array of exactly 5 partial sums to be combined
 */
void softmax_calculate_inv_sum(float *inv_sum, const float sums[]);

/**
 * @brief Generate the 256-entry look-up table of exponentials used by softmax_exp_sum() and softmax_exp_div().
 *
 * @param[in]   zero_point  Quantization zero-point of the softmax input
 * @param[in]   scale       Quantization scale of the softmax input
 * @param[out]  lut         The generated look-up table, with 256 `float32` entries
 */
void softmax_generate_exp_lut(int zero_point, float scale, float *lut);

/**
 * @brief Compute softmax for a single vector.
 *
 * @param[out]  Y           The output vector
 * @param[in]   X           The input vector
 * @param[in]   zero_point  Quantization zero-point of the input
 * @param[in]   scale       Quantization scale of the input
 * @param[in]   length      Number of elements in the input and output vectors
 */
void softmax(int8_t *Y, const int8_t *X, const float zero_point,
             const float scale, const int length);

/**
 * @brief Compute softmax for a single vector using a precomputed exponential look-up table.
 *
 * @param[out]  Y       The output vector
 * @param[in]   X       The input vector
 * @param[in]   lut     Look-up table of exponentials (see softmax_generate_exp_lut())
 * @param[in]   offset  Number of elements in the input and output vectors
 */
void softmax_single(int8_t *Y, const int8_t *X, const float *lut,
                    const int offset);

/**
 * @brief Compute the mean, over a middle dimension, of an 8-bit tensor.
 *
 * The input is treated as a 3D tensor of shape (`start_dim_size`, `mean_dim_size`, `end_dim_size`); the mean is computed over the middle (`mean_dim_size`) dimension, producing an output of shape (`start_dim_size`, `end_dim_size`).
 *
 * @param[in]   input           The input tensor
 * @param[out]  output          The output tensor
 * @param[in]   start_dim_size  Size of the outermost dimension
 * @param[in]   mean_dim_size   Size of the dimension being averaged over
 * @param[in]   end_dim_size    Size of the innermost dimension
 * @param[in]   in_zero_point   Quantization zero-point of the input
 * @param[in]   out_zero_point  Quantization zero-point of the output
 * @param[in]   scale_mul       Scale factor applied to the computed mean
 */
void mean_int8(const int8_t *input, int8_t *output, const int start_dim_size,
               const int mean_dim_size, const int end_dim_size,
               const float in_zero_point, const float out_zero_point,
               const float scale_mul);

/**
 * @brief Compute the mean, over a middle dimension, of a 16-bit tensor.
 *
 * See mean_int8() for a description of how the input tensor's dimensions relate to the output.
 *
 * @param[in]   input           The input tensor
 * @param[out]  output          The output tensor
 * @param[in]   start_dim_size  Size of the outermost dimension
 * @param[in]   mean_dim_size   Size of the dimension being averaged over
 * @param[in]   end_dim_size    Size of the innermost dimension
 * @param[in]   scale_mul       Scale factor applied to the computed mean
 */
void mean_int16(const int16_t *input, int16_t *output, const int start_dim_size,
                const int mean_dim_size, const int end_dim_size,
                const float scale_mul);

/**
 * @brief Return the index of the maximum value in an int16 vector.
 *
 * @param[out]  output_index  The index of the maximum input value
 * @param[in]   input_values  The input int16 vector
 * @param[in]   element_count The number of input values
 */
void argmax_16(int32_t *output_index, const int16_t *input_values,
               const int32_t element_count);

#endif // LAYERS_H_
