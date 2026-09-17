// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * @file nn_layers.h
 * @brief Layer Operations for neural network.
 */

#pragma once

#include "nn_api.h"
#include "nn_bin_types.h"
#include "nn_image.h"
#include <string.h>

#include "output_transform_fn_int16_mappings.h"

// Defines
#define ADD_INT16_TENSOR_BYTES() (2 * 16 * sizeof(int16_t))
#define DEQUANTIZE_INT16_TENSOR_BYTES() (2 * sizeof(float))
#define MULTIPLY_INT16_TENSOR_BYTES() (2 * sizeof(int16_t))
#define QUANTIZE_INT16_TENSOR_BYTES() (1 * sizeof(float))
#define REQUANTIZE_INT16_TENSOR_BYTES() (16 * sizeof(int16_t))
#define QUADRATIC_APPROXIMATION_MAX_CHUNKS (2048)

#ifdef __xcore__
#define ACTIVATION_FUNCTION __attribute__((fptrgroup("activation_functions")))
#else
#define ACTIVATION_FUNCTION /**/
#endif

// Structs
/** @brief Parameters for the signed 16-bit output transform. */
typedef struct {
  int32_t output_slice_channel_count; ///< Number of channels to transform.
} otfn_int16_params_t;

/**
 * @brief Store a quadratic approximation table.
 *
 * On XS3, the table must be 64-bit aligned when passed to assembly code.
 */
struct quadratic_function_table {
  struct { // Field order is part of the assembly interface.
    int32_t c;
    int8_t a;
    int8_t padding;
    int16_t b;
  } coefficients[QUADRATIC_APPROXIMATION_MAX_CHUNKS]; ///< Approximation
                                                      ///< coefficients.
  int data_bytes; ///< Number of populated coefficient bytes.
};

typedef struct quadratic_function_table quadratic_function_table_t;

/** @brief Function pointer from one floating-point value to another. 
 * Used by @ref quadratic_approximation_generator().
*/
typedef float (*float_function_t)(float x);

/**
 * @brief Describe the contiguous input range processed by one bsign_8() job.
 * @note This struct is intended to be opaque.
 */
typedef struct {
  mem_stride_t start; ///< Internal job start offset.
  int32_t length;     ///< Internal job length.
} nn_bsign_8_job_t;

/** @brief Store transformed parameters for mul_elementwise(). */
typedef struct nn_mul_params_t {
  int8_t in1_zero_point; ///< Internal transformed parameter.
  int8_t in2_zero_point; ///< Internal transformed parameter.
  int16_t bias;          ///< Internal transformed parameter.
  int16_t scalar;        ///< Internal transformed parameter.
  int16_t vlashr_shr;    ///< Internal transformed parameter.
} nn_mul_params_t;

/** @brief Store transformed parameters for add_elementwise(). */
typedef struct {
  int16_t m1[16];      ///< Internal transformed parameters.
  int16_t m2[16];      ///< Internal transformed parameters.
  int16_t shift[16];   ///< Internal transformed parameters.
  int16_t bias_hi[16]; ///< Internal transformed parameters.
  int16_t bias_lo[16]; ///< Internal transformed parameters.
} nn_add_params_t;

/**
 * @brief Store quantization and shape parameters for mat_mul_real_int8().
 */
typedef struct {
  float lhs_zp;          ///< Left-hand side zero point.
  float rhs_zp;          ///< Right-hand side zero point.
  float in_zp_sum;       ///< `channel_size * lhs_zp * rhs_zp`.
  float out_zp;          ///< Output zero point.
  float scale;           ///< `lhs_scale * rhs_scale / output_scale`.
  uint32_t lhs_row_size; ///< Number of left-hand side rows.
  uint32_t channel_size; ///< Shared inner matrix dimension.
  uint32_t rhs_col_size; ///< Number of right-hand side columns.
} nn_mat_mul_real_params_t;

// Functions
/**
 * @brief Generate the transformed parameters used by quantize_int16_tensor().
 *
 * Call this at build time and pass the output blob to quantize_int16_tensor()
 * at run time.
 *
 * @param[out] output        Output blob of QUANTIZE_INT16_TENSOR_BYTES() bytes
 * @param[in]  output_scaler Quantization scale of the output tensor
 * @return 1 on success, or 0 when a fallback implementation is required.
 * @note On XS3, `output` must be word-aligned.
 */
C_API int quantize_int16_tensor_blob(void *output, float output_scaler);

/**
 * @brief Quantize a float tensor into an int16 tensor.
 *
 * `blob` must have been created by quantize_int16_tensor_blob().
 *
 * @param[out] output        Output tensor
 * @param[in]  input         Input tensor
 * @param[in]  tensor_length Number of elements in the tensor (product of all
 * dimensions)
 * @param[in]  blob          Transformed quantization parameters
 * @note On XS3, `output`, `input`, and `blob` must be word-aligned.
 */
C_API void quantize_int16_tensor(int16_t *output, float *input,
                                 int tensor_length, void *blob);

/**
 * @brief Generate the transformed parameters used by requantize_int16_tensor().
 *
 * Call this at build time and pass the output blob to requantize_int16_tensor()
 * at run time.
 *
 * @param[out] output        Output blob of REQUANTIZE_INT16_TENSOR_BYTES()
 * bytes
 * @param[in]  input_scaler  Quantization scale of the input tensor
 * @param[in]  output_scaler Quantization scale of the output tensor
 * @param[out] err_msg       Error message populated when the transformation
 * fails
 * @return 1 on success, or 0 when a fallback implementation is required.
 * @note On XS3, `output` must be word-aligned.
 */
C_API int requantize_int16_tensor_blob(void *output, float input_scaler,
                                       float output_scaler, char *err_msg);

/**
 * @brief Requantize an int16 tensor into an int16 tensor.
 *
 * `blob` must have been created by requantize_int16_tensor_blob().
 *
 * @param[out] output        Output tensor
 * @param[in]  input         Input tensor
 * @param[in]  tensor_length Number of elements in the tensor (product of all
 * dimensions)
 * @param[in]  blob          Transformed quantization parameters
 * @note On XS3, `output`, `input`, and `blob` must be word-aligned.
 */
C_API void requantize_int16_tensor(int16_t *output, int16_t *input,
                                   int tensor_length, void *blob);

/**
 * @brief Requantize a range of int16 elements into int8 elements.
 *
 * `elm_start` and `elm_count` together specify the output elements computed by
 * this invocation, namely those for which `elm_start <= k < elm_start +
 * elm_count`.
 *
 * @param[out] y         Output vector
 * @param[in]  x         Input vector
 * @param[in]  elm_start Index of the first output element to compute
 * @param[in]  elm_count Number of output elements to compute
 */
C_API void requantize_16_to_8(int8_t *y, const int16_t *x,
                              const unsigned elm_start,
                              const unsigned elm_count);

/**
 * @brief Generate the transformed parameters used by multiply_int16_tensor().
 *
 * Call this at build time and pass the output blob to multiply_int16_tensor()
 * at run time.
 *
 * @param[out] output        Output blob of MULTIPLY_INT16_TENSOR_BYTES() bytes
 * @param[in]  input1_scaler Quantization scale of the first input tensor
 * @param[in]  input2_scaler Quantization scale of the second input tensor
 * @param[in]  output_scaler Quantization scale of the output tensor
 * @param[out] err_msg       Error message populated when the transformation
 * fails
 * @return 1 on success, or 0 when a fallback implementation is required.
 * @note On XS3, `output` must be word-aligned.
 */
C_API int multiply_int16_tensor_blob(void *output, float input1_scaler,
                                     float input2_scaler, float output_scaler,
                                     char *err_msg);

/**
 * @brief Multiply two int16 tensors into an int16 tensor.
 *
 * `blob` must have been created by multiply_int16_tensor_blob().
 *
 * @param[out] output        Output tensor
 * @param[in]  input1        First input tensor
 * @param[in]  input2        Second input tensor
 * @param[in]  tensor_length Number of elements in each tensor
 * @param[in]  blob          Transformed quantization parameters
 * @note On XS3, `output`, `input1`, `input2`, and `blob` must be word-aligned.
 */
C_API void multiply_int16_tensor(int16_t *output, int16_t *input1,
                                 int16_t *input2, int tensor_length,
                                 void *blob);

/**
 * @brief Expand a signed 8-bit vector into a signed 16-bit vector.
 *
 * Each input element is sign-extended to 16 bits and stored in the
 * corresponding output element. Only the first `N` elements of `out` are
 * written.
 *
 * @param[out] out Output vector with space for at least `N` int16_t elements.
 * @param[in]  in  Input vector with at least `N` int8_t elements.
 * @param[in]  N   Number of elements to expand.
 */
C_API void expand_8_to_16(int16_t *out, int8_t *in, int N);

/**
 * @brief Add two int16 tensors using transformed quantization parameters.
 *
 * @param[out] output        Output tensor
 * @param[in]  input1        First input tensor
 * @param[in]  input2        Second input tensor
 * @param[in]  tensor_length Number of elements in each tensor
 * @param[in]  blob          Parameters generated by @ref add_int16_tensor_blob()
 * @note On XS3, `output`, `input1`, `input2`, and `blob` must be word-aligned.
 */
C_API void add_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2,
                            int tensor_length, void *blob);

/**
 * @brief Generate the transformed parameters used by add_int16_tensor().
 *
 * @param[out] output        Output blob of ADD_INT16_TENSOR_BYTES() bytes
 * @param[in]  input1_scaler Quantization scale of the first input tensor
 * @param[in]  input2_scaler Quantization scale of the second input tensor
 * @param[in]  output_scaler Quantization scale of the output tensor
 * @param[out] err_msg       Error message populated when the transformation fails
 * @return 1 on success, or 0 when a fallback implementation is required.
 * @note On XS3, `output` must be word-aligned.
 */
C_API int add_int16_tensor_blob(void *output, float input1_scaler,
                                float input2_scaler, float output_scaler,
                                char *err_msg);


/**
 * @brief Generate the constant parameters used to dequantize an int16 tensor.
 *
 * Call this at build time and pass the output blob to
 * dequantize_int16_tensor() at run time.
 *
 * @param[out] output       Output blob of DEQUANTIZE_INT16_TENSOR_BYTES() bytes
 * @param[in]  input_scaler Quantization scale of the input tensor
 * @param[out] err_msg      Error message populated when the transformation
 * fails
 * @return 1 on success, or 0 when a fallback implementation is required.
 * @note On XS3, `output` must be word-aligned.
 */
C_API int dequantize_int16_tensor_blob(void *output, float input_scaler,
                                       char *err_msg);

/**
 * @brief Dequantize an int16 tensor into a float tensor.
 *
 * `blob` must have been created by dequantize_int16_tensor_blob().
 *
 * @param[out] output        Output tensor
 * @param[in]  input         Input tensor
 * @param[in]  tensor_length Number of elements in the tensor
 * @param[in]  blob          Transformed dequantization parameters
 * @note On XS3, `output` and `input` must be word-aligned.
 */
C_API void dequantize_int16_tensor(float *output, int16_t *input,
                                   int tensor_length, void *blob);

/**
 * @brief Find the index of the maximum value in an int16 vector.
 *
 * @param[out] idx Output index of the maximum value.
 * @param[in]  X Input int16 vector.
 * @param[in]  N Number of elements in the input vector.
 */
C_API void argmax_16(int32_t *idx, const int16_t *X, const int32_t N);

/**
 * @brief Prepare jobs and the threshold vector used by bsign_8().
 *
 * Splits `N` elements into disjoint jobs and fills `zero_point_vect` with 32
 * copies of `zero_point`. Jobs may be run in parallel.
 *
 * @param[out] jobs             Array of `job_count` jobs to initialize
 * @param[out] zero_point_vect  32-element threshold vector
 * @param[in]  N                Number of input elements
 * @param[in]  zero_point       Comparison threshold
 * @param[in]  job_count        Number of jobs to initialize; must be positive
 */
C_API void bsign_8_prepare(nn_bsign_8_job_t *jobs, int8_t *zero_point_vect,
                           const uint32_t N, const int8_t zero_point,
                           const int32_t job_count);

/**
 * @brief Compare int8 elements with a threshold and pack the results.
 *
 * Packs 32 comparisons per word, least-significant bit first. A bit is one
 * when `X[i] < zero_point` and zero otherwise; unused tail bits are zero.
 * `X`, `Y`, `job`, and `zero_point_vect` refer to the complete operation.
 *
 * @param[out] Y               Packed output vector
 * @param[in]  X               Complete int8 input vector
 * @param[in]  zero_point_vect Threshold vector from bsign_8_prepare()
 * @param[in]  job             Prepared input range to process
 * @note On XS3, `Y` and `X` must be word-aligned.
 */
C_API void bsign_8(bnn_b32_t *Y, const int8_t *X, const int8_t *zero_point_vect,
                   const nn_bsign_8_job_t *job);

/**
 * @brief Compute the number of 3-byte blocks for pad_3_to_4_run().
 *
 * @param[out]  n_3     Number of 3-byte blocks
 * @param[in]   height  Image height, in pixels
 * @param[in]   width   Image width, in pixels
 */
C_API void pad_3_to_4_prepare(uint32_t *n_3, const unsigned height,
                              const unsigned width);

/**
 * @brief Pad 3-byte pixels to 4 bytes, setting the added byte to a specified
 * value.
 *
 * @param[out] outputs Output pixels containing three input bytes and one pad
 * byte
 * @param[in]  inputs  Input bytes, for example RGBRGBRGBRGB
 * @param[in]  N_3     Number of 3-byte blocks to copy
 * @param[in]  pad_val Value written to each padding byte
 * @note On XS3, `outputs` must be word-aligned.
 */
C_API void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3,
                          uint32_t pad_val);

/**
 * @brief Pad individual bytes into 32-bit words.
 *
 * Each input byte becomes the least-significant byte of a word. The function
 * processes `N * 4` input bytes, so `N` counts 4-byte input chunks.
 *
 * @param[out] outputs Output words containing one input byte and three pad
 * bytes
 * @param[in]  inputs  Input bytes
 * @param[in]  N       Number of 4-byte input chunks
 * @param[in]  pad_val Value written to the upper three bytes of each output
 * word
 */
C_API void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N,
                          uint32_t pad_val);

/**
 * @brief Compute quantization parameters for mul_elementwise().
 *
 * @param[out]  params        The computed parameters
 * @param[in]   in1Scale      Quantization scale of the first input
 * @param[in]   in2Scale      Quantization scale of the second input
 * @param[in]   outputScale   Quantization scale of the output
 * @param[in]   in1ZeroPoint  Quantization zero-point of the first input
 * @param[in]   in2ZeroPoint  Quantization zero-point of the second input
 * @param[in]   outputZeroPoint  Quantization zero-point of the output
 */
C_API void mul_boggle(nn_mul_params_t *params, double in1Scale, double in2Scale,
                      double outputScale, int8_t in1ZeroPoint,
                      int8_t in2ZeroPoint, int8_t outputZeroPoint);

/**
 * @brief Multiply two quantized int8 vectors element by element.
 *
 * `params` must have been populated by mul_boggle().
 *
 * @param[in]   in1_data       The first input vector
 * @param[in]   in2_data       The second input vector
 * @param[in]   element_count  Number of elements to compute
 * @param[in]   params         The quantization parameters
 * @param[out]  out_data       The output vector
 */
C_API void mul_elementwise(const int8_t *in1_data, const int8_t *in2_data,
                           int element_count, nn_mul_params_t *params,
                           int8_t *out_data);

/**
 * @brief Add two quantized int8 vectors element by element.
 *
 * The function computes elements from `start` through
 * `start + count - 1`.
 *
 * @param[out]  Y           The output vector
 * @param[in]   X1          The first input vector
 * @param[in]   X2          The second input vector
 * @param[in]   p           The scaling and bias parameters
 * @param[in]   start       Index of first output element to be computed
 * @param[in]   count       Number of output elements to be computed
 */
C_API void add_elementwise(int8_t Y[], const int8_t X1[], const int8_t X2[],
                           nn_add_params_t *p, const int start,
                           const int count);

/**
 * @brief Apply an 8-bit look-up table to a vector, element-by-element.
 *
 * `Y` and `X` must point to the starts of their complete vectors.
 *
 * @param[out] Y         Output vector
 * @param[in]  X         Input vector
 * @param[in]  lut       Look-up table with 256 uint8_t entries
 * @param[in]  elm_start Index of first output element to compute
 * @param[in]  elm_count Number of output elements to compute
 * @note On XS3, `Y` and `X` must be word-aligned.
 */
C_API void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
                   const unsigned elm_start, const unsigned elm_count);

/**
 * @brief Sum the exponentials of a range of elements of a softmax input vector.
 *
 * `lut` maps all 256 int8 values to exponentials. The function sums the range
 * selected by `elm_start` and `elm_count`.
 *
 * @param[out] Y         Sum of exponentials
 * @param[in]  X         Input vector
 * @param[in]  lut       Look-up table of 256 exponentials
 * @param[in]  elm_start Index of first input element to sum
 * @param[in]  elm_count Number of input elements to sum
 */
C_API void softmax_exp_sum(float *Y, const int8_t *X, const float *lut,
                           const unsigned elm_start, const unsigned elm_count);

/**
 * @brief Produce softmax outputs from a precomputed exponential table.
 *
 * `inv_sum` is 256 divided by the sum of all exponentials.
 *
 * @param[out] Y         Output vector
 * @param[in]  X         Input vector
 * @param[in]  lut       Look-up table of 256 exponentials
 * @param[in]  inv_sum   256 divided by the total exponential sum
 * @param[in]  elm_start Index of first output element to compute
 * @param[in]  elm_count Number of output elements to compute
 */
C_API void softmax_exp_div(int8_t *Y, const int8_t *X, const float *lut,
                           const float inv_sum, const unsigned elm_start,
                           const unsigned elm_count);

/**
 * @brief Compute the reciprocal of the sum of a set of partial sums.
 *
 * The function reads exactly five partial sums and computes 256 divided by
 * their total.
 *
 * @param[out] inv_sum 256 divided by the total sum
 * @param[in]  sums    Array of five partial sums
 */
C_API void softmax_calculate_inv_sum(float *inv_sum, const float sums[]);

/**
 * @brief Generate the 256-entry exponential table used by softmax.
 *
 * @param[in]   zero_point  Quantization zero-point of the softmax input
 * @param[in]   scale       Quantization scale of the softmax input
 * @param[out]  lut         The generated look-up table, with 256 `float32`
 * entries
 */
C_API void softmax_generate_exp_lut(int zero_point, float scale, float *lut);

/**
 * @brief Compute softmax for a single vector.
 *
 * @param[out]  Y           The output vector
 * @param[in]   X           The input vector
 * @param[in]   zero_point  Quantization zero-point of the input
 * @param[in]   scale       Quantization scale of the input
 * @param[in]   length      Number of elements in the input and output vectors
 */
C_API void softmax(int8_t *Y, const int8_t *X, const float zero_point,
                   const float scale, const int length);

/**
 * @brief Compute softmax using a precomputed exponential table.
 *
 * @param[out]  Y       The output vector
 * @param[in]   X       The input vector
 * @param[in]   lut     Look-up table of exponentials (see
 * softmax_generate_exp_lut())
 * @param[in]   offset  Number of elements in the input and output vectors
 */
C_API void softmax_single(int8_t *Y, const int8_t *X, const float *lut,
                          const int offset);

/**
 * @brief Compute the mean, over a middle dimension, of an 8-bit tensor.
 *
 * The input shape is (`start_dim_size`, `mean_dim_size`, `end_dim_size`). The
 * output shape is (`start_dim_size`, `end_dim_size`).
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
C_API void mean_int8(const int8_t *input, int8_t *output,
                     const int start_dim_size, const int mean_dim_size,
                     const int end_dim_size, const float in_zero_point,
                     const float out_zero_point, const float scale_mul);

/**
 * @brief Compute the mean, over a middle dimension, of a 16-bit tensor.
 *
 * See mean_int8() for a description of how the input tensor's dimensions relate
 * to the output.
 *
 * @param[in]   input           The input tensor
 * @param[out]  output          The output tensor
 * @param[in]   start_dim_size  Size of the outermost dimension
 * @param[in]   mean_dim_size   Size of the dimension being averaged over
 * @param[in]   end_dim_size    Size of the innermost dimension
 * @param[in]   scale_mul       Scale factor applied to the computed mean
 */
C_API void mean_int16(const int16_t *input, int16_t *output,
                      const int start_dim_size, const int mean_dim_size,
                      const int end_dim_size, const float scale_mul);

/**
 * @brief Multiply real int8 matrices using quantization parameters.
 *
 * @param[in]   p         The scaling and bias parameters
 * @param[out]  vpu_buf0  Temporary VPU buffer, length 64
 * @param[out]  vpu_buf1  Temporary VPU buffer, length 64
 * @param[in]   lhs       The left-hand side matrix, row major
 * @param[in]   rhs       The right-hand side matrix, column major
 * @param[out]  output    The output matrix
 */
C_API void mat_mul_real_int8(nn_mat_mul_real_params_t *p, int8_t *vpu_buf0,
                             int8_t *vpu_buf1, int8_t *lhs, int8_t *rhs,
                             int8_t *output);

/**
 * @brief Prepare per-channel parameters for the signed int16 output transform.
 *
 * Multipliers are converted to Q2.30. Each 16-channel group is packed as:
 *
 * \code
 * a1, a3, ... a15, m1, m3, ... m15,
 * a0, a2, ... a14, m0, m2, ... m14
 * \endcode
 *
 * @param[in]  kernel_weights_in      Unused
 * @param[in]  channel_multipliers_in Per-channel floating-point multipliers
 * @param[in]  channel_bias_terms_in  Per-channel accumulator-domain biases
 * @param[out] kernel_weights_out     Unused
 * @param[out] mul_add_out            Packed buffer with 32 elements per channel
 * group
 * @param[in]  input_channels         Unused
 * @param[in]  output_channels        Number of output channels to prepare
 */
C_API void output_transform_fn_int16_kernel_transform(
    const int8_t *kernel_weights_in, const float *channel_multipliers_in,
    const int *channel_bias_terms_in, int8_t *kernel_weights_out,
    int32_t *mul_add_out, int input_channels, int output_channels);

/**
 * @brief Transform up to 16 accumulators into signed int16 outputs.
 *
 * Each accumulator is reconstructed from its low half in vR and its high half
 * in vD. Bias addition and Q2.30 multiplication use signed saturation before
 * the result is saturated to int16. `vDvR` contains 16 vR values followed by
 * 16 vD values. `output` must provide space for 16 values.
 *
 * @param[in]  params               Output slice parameters and channel count
 * @param[out] output               Destination for the current output group
 * @param[in]  vDvR                 32-element accumulator buffer
 * @param[in]  output_channel_group Zero-based group of 16 output channels
 * @param[in]  mul_add              Packed biases and Q2.30 multipliers
 * @return Pointer immediately after the last output value written.
 * @note On XS3, `vDvR` must be eight-byte aligned and `output` must be
 * word-aligned.
 */
C_API int16_t *output_transform_fn_int16(otfn_int16_params_t *params,
                                         int16_t *output, int16_t *vDvR,
                                         int32_t output_channel_group,
                                         int32_t *mul_add);

/**
 * @brief Build a quadratic approximation table for a monotonic function.
 *
 * The assembly interpolation implementation requires 128 chunks.
 *
 * @param[out] table         Approximation table to populate
 * @param[in]  av            Function to approximate
 * @param[in]  input_scaler  Scale applied to the input
 * @param[in]  output_scaler Scale applied to the output
 * @param[in]  chunks        Number of interpolation chunks
 * @param[out] max_error     Maximum approximation error
 * @param[out] error         Square root of the sum of squared errors
 */
C_API void
quadratic_approximation_generator(quadratic_function_table_t *table,
                                  ACTIVATION_FUNCTION float_function_t av,
                                  double input_scaler, double output_scaler,
                                  int chunks, int *max_error, double *error);

/**
 * @brief Return the number of bytes used by an approximation table.
 *
 * @param[in] x Approximation table
 * @return Number of bytes in the table.
 */
C_API uint32_t
quadratic_function_table_number_bytes(quadratic_function_table_t *x);

/**
 * @brief Return the byte representation of an approximation table.
 *
 * @param[in] x Approximation table
 * @return Pointer to the table bytes.
 */
C_API uint8_t *quadratic_function_table_bytes(quadratic_function_table_t *x);

/**
 * @brief Evaluate the hyperbolic tangent activation function.
 *
 * Computes `tanh(x)`, producing a value in the range `[-1, 1]`.
 *
 * @param[in] x Real-valued input
 * @return The hyperbolic tangent of `x`.
 */
C_API float approximation_function_tanh(float x);

/**
 * @brief Evaluate the logistic sigmoid activation function.
 *
 * Computes `1 / (1 + exp(-x))`, producing a value in the range `[0, 1]`.
 *
 * @param[in] x Real-valued input
 * @return The logistic sigmoid of `x`.
 */
C_API float approximation_function_logistics(float x);

/**
 * @brief Evaluate the exponential linear unit (ELU) activation function.
 *
 * Uses `alpha = 1`: returns `x` when `x >= 0`, otherwise `exp(x) - 1`.
 * The result is in the range `(-1, infinity)`.
 *
 * @param[in] x Real-valued input
 * @return The ELU activation of `x`.
 */
C_API float approximation_function_elu(float x);

/**
 * @brief Evaluate the rectified linear unit (ReLU) activation function.
 *
 * Computes `max(0, x)`, producing a non-negative result.
 *
 * @param[in] x Real-valued input
 * @return Zero when `x <= 0`; otherwise, `x`.
 */
C_API float approximation_function_relu(float x);

/**
 * @brief Evaluate the ReLU6 activation function.
 *
 * Computes `min(max(0, x), 6)`, producing a value in the range `[0, 6]`.
 *
 * @param[in] x Real-valued input
 * @return Zero when `x <= 0`, six when `x >= 6`; otherwise, `x`.
 */
C_API float approximation_function_relu6(float x);

/**
 * @brief Apply a 128-chunk quadratic interpolation to an int16 vector.
 *
 * @param[out] outputs Output vector
 * @param[in]  inputs  Input vector
 * @param[in]  coeffs  Table from quadratic_approximation_generator()
 * @param[in]  N       Number of elements in the vectors
 * @note On XS3, `coeffs` must be 64-bit aligned.
 */
C_API void quadratic_interpolation_128(int16_t *outputs, int16_t *inputs,
                                       uint8_t *coeffs, uint32_t N);
