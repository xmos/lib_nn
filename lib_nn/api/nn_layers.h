// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#include <stdint.h>

#include "nn_api.h"
#include "nn_bin_types.h"
#include "nn_image.h"

// ---------- defines ----------

#if defined(__xcore__) || defined(__riscv_xxcore)
#define ACTIVATION_FUNCTION __attribute__((fptrgroup("activation_functions")))
#else
#define ACTIVATION_FUNCTION
#endif

#define QUADRATIC_APPROXIMATION_MAX_CHUNKS      (2048)
#define ADD_INT16_TENSOR_BYTES()                (2 * 16 * sizeof(int16_t))
#define DEQUANTIZE_INT16_TENSOR_BYTES()         (2 * sizeof(float))
#define QUANTIZE_INT16_TENSOR_BYTES()           (sizeof(float))
#define REQUANTIZE_INT16_TENSOR_BYTES()         (16 * sizeof(int16_t))
#define MULTIPLY_INT16_TENSOR_BYTES()           (2 * sizeof(int16_t))

// ---------- structs ----------

/** Parameters for one bsign_8 job. */
typedef struct {
  mem_stride_t start;
  int32_t length;
} nn_bsign_8_job_t;

/** Parameters used by mul_elementwise(). */
typedef struct nn_mul_params_t {
  int8_t in1_zero_point;
  int8_t in2_zero_point;
  int16_t bias;
  int16_t scalar;
  int16_t vlashr_shr;
} nn_mul_params_t;

/** Parameters used by add_elementwise(). */
typedef struct {
  int16_t m1[16];
  int16_t m2[16];
  int16_t shift[16];
  int16_t bias_hi[16];
  int16_t bias_lo[16];
} nn_add_params_t;

/** Parameters for the int16 output transform. */
typedef struct {
  int32_t output_slice_channel_count;
} otfn_int16_params_t;

typedef struct quadratic_function_table {
  struct {
    int32_t c;
    int8_t a;
    int8_t padding;
    int16_t b;
  } coefficients[QUADRATIC_APPROXIMATION_MAX_CHUNKS];
  int data_bytes;
} quadratic_function_table_t;

typedef float (*float_function_t)(float x);

// ---------- functions ----------

/**
 * @brief Initialize jobs used by bsign_8().
 * @param[out] jobs Array of jobs to initialize.
 * @param[out] zero_point_vect Per-channel zero-point vector.
 * @param[in] length Number of scalar input elements.
 * @param[in] zero_point Input zero point used for padding.
 * @param[in] job_count Number of jobs to initialize.
 */
C_API void bsign_8_prepare(nn_bsign_8_job_t *jobs, int8_t *zero_point_vect,
                     uint32_t length, int8_t zero_point, int32_t job_count);

/**
 * @brief Compute the bit-packed sign of each element in an int8 vector.
 * @param[out] Y Bit-packed output vector; must be word-aligned.
 * @param[in] X Input vector; must be word-aligned.
 * @param[in] zero_point_vect Per-channel zero-point vector.
 * @param[in] job Job describing the input range to process.
 */
C_API void bsign_8(bnn_b32_t *Y, const int8_t *X, const int8_t *zero_point_vect,
             const nn_bsign_8_job_t *job);

/**
 * @brief Compute the number of 3-byte blocks required by pad_3_to_4_run().
 * @param[out] n_3 Receives the number of 3-byte blocks.
 * @param[in] height Image height in pixels.
 * @param[in] width Image width in pixels.
 */
C_API void pad_3_to_4_prepare(uint32_t *n_3, unsigned height, unsigned width);

/**
 * @brief Pad 3-byte pixels to 4 bytes per pixel.
 * @param[out] outputs Word-aligned padded output image.
 * @param[in] inputs Input image containing 3-byte pixels.
 * @param[in] N_3 Number of 3-byte pixels to process.
 * @param[in] pad_val Value used for the padding byte.
 */
C_API void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3,
                    uint32_t pad_val);

/**
 * @brief Expand byte values into 32-bit words with three padding bytes.
 * @param[out] outputs Padded output buffer.
 * @param[in] inputs Input byte buffer.
 * @param[in] N Number of four-byte input chunks.
 * @param[in] pad_val Value used for the three padding bytes.
 */
C_API void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N,
                    uint32_t pad_val);

/**
 * @brief Compute parameters for mul_elementwise().
 * @param[out] params Quantization parameters for the multiplication.
 * @param[in] in1_scale First input scale.
 * @param[in] in2_scale Second input scale.
 * @param[in] output_scale Output scale.
 * @param[in] in1_zero_point First input zero point.
 * @param[in] in2_zero_point Second input zero point.
 * @param[in] output_zero_point Output zero point.
 */
C_API void mul_boggle(nn_mul_params_t *params, double in1_scale, double in2_scale,
                double output_scale, int8_t in1_zero_point,
                int8_t in2_zero_point, int8_t output_zero_point);

/**
 * @brief Multiply two quantized int8 vectors element by element.
 * @param[in] in1_data First input vector.
 * @param[in] in2_data Second input vector.
 * @param[in] element_count Number of elements to compute.
 * @param[in] params Quantization parameters from mul_boggle().
 * @param[out] out_data Output vector.
 */
C_API void mul_elementwise(const int8_t *in1_data, const int8_t *in2_data,
                     int element_count, nn_mul_params_t *params,
                     int8_t *out_data);

/**
 * @brief Add two quantized int8 vectors element by element.
 * @param[out] Y Output vector.
 * @param[in] X1 First input vector.
 * @param[in] X2 Second input vector.
 * @param[in] params Scaling and bias parameters.
 * @param[in] output_start First output element to compute.
 * @param[in] output_count Number of output elements to compute.
 */
C_API void add_elementwise(int8_t Y[], const int8_t X1[], const int8_t X2[],
                     nn_add_params_t *params, int output_start,
                     int output_count);

/**
 * @brief Apply an 8-bit lookup table to a vector.
 * @param[out] Y Output vector.
 * @param[in] X Input vector.
 * @param[in] lut 256-entry lookup table.
 * @param[in] elm_start First element to process.
 * @param[in] elm_count Number of elements to process.
 */
C_API void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
             unsigned elm_start, unsigned elm_count);

/**
 * @brief Generate the exponential lookup table used by the softmax helpers.
 * @param[in] zero_point Quantization zero point.
 * @param[in] scale Quantization scale.
 * @param[out] lut Output 256-entry exponential table.
 */
C_API void softmax_generate_exp_lut(int zero_point, float scale, float *lut);

/**
 * @brief Sum exponentials for a range of softmax input elements.
 * @param[out] Y Output partial sum.
 * @param[in] X Input vector.
 * @param[in] lut Exponential lookup table.
 * @param[in] elm_start First input element to sum.
 * @param[in] elm_count Number of input elements to sum.
 */
C_API void softmax_exp_sum(float *Y, const int8_t *X, const float *lut,
                     unsigned elm_start, unsigned elm_count);

/**
 * @brief Divide exponentials by the complete softmax sum for a range of elements.
 * @param[out] Y Output vector.
 * @param[in] X Input vector.
 * @param[in] lut Exponential lookup table.
 * @param[in] inv_sum Reciprocal sum scaling factor.
 * @param[in] elm_start First output element to compute.
 * @param[in] elm_count Number of output elements to compute.
 */
C_API void softmax_exp_div(int8_t *Y, const int8_t *X, const float *lut,
                     float inv_sum, unsigned elm_start, unsigned elm_count);

/**
 * @brief Combine five partial sums into the softmax scaling factor.
 * @param[out] inv_sum Output reciprocal sum scaling factor.
 * @param[in] sums Five partial sums.
 */
C_API void softmax_calculate_inv_sum(float *inv_sum, const float sums[]);

/**
 * @brief Compute softmax for one vector.
 * @param[out] Y Output vector.
 * @param[in] X Input vector.
 * @param[in] zero_point Quantization zero point.
 * @param[in] scale Quantization scale.
 * @param[in] length Number of input and output elements.
 */
C_API void softmax(int8_t *Y, const int8_t *X, float zero_point, float scale,
             int length);

/**
 * @brief Compute softmax using a precomputed exponential lookup table.
 * @param[out] Y Output vector.
 * @param[in] X Input vector.
 * @param[in] lut Exponential lookup table.
 * @param[in] offset Number of input and output elements.
 */
C_API void softmax_single(int8_t *Y, const int8_t *X, const float *lut, int offset);

/**
 * @brief Compute the mean over the middle dimension of an int8 tensor.
 * @param[in] input Input tensor.
 * @param[out] output Output tensor.
 * @param[in] start_dim_size Size of the outer dimension.
 * @param[in] mean_dim_size Size of the dimension to average.
 * @param[in] end_dim_size Size of the inner dimension.
 * @param[in] in_zero_point Input zero point.
 * @param[in] out_zero_point Output zero point.
 * @param[in] scale_mul Mean scaling factor.
 */
C_API void mean_int8(const int8_t *input, int8_t *output, int start_dim_size,
              int mean_dim_size, int end_dim_size, float in_zero_point,
              float out_zero_point, float scale_mul);

/**
 * @brief Compute the mean over the middle dimension of an int16 tensor.
 * @param[in] input Input tensor.
 * @param[out] output Output tensor.
 * @param[in] start_dim_size Size of the outer dimension.
 * @param[in] mean_dim_size Size of the dimension to average.
 * @param[in] end_dim_size Size of the inner dimension.
 * @param[in] scale_mul Mean scaling factor.
 */
C_API void mean_int16(const int16_t *input, int16_t *output, int start_dim_size,
               int mean_dim_size, int end_dim_size, float scale_mul);

/**
 * @brief Return the index of the maximum value in an int16 vector.
 * @param[out] output_index Index of the maximum value.
 * @param[in] input_values Input vector.
 * @param[in] element_count Number of input elements.
 */
C_API void argmax_16(int32_t *output_index, const int16_t *input_values,
               int32_t element_count);

/**
 * @brief Transform an int16 accumulator into an int16 output vector.
 * @param[in] params Output-transform parameters.
 * @param[out] output Output vector.
 * @param[in] vDvR Accumulator ring-buffer contents.
 * @param[in] output_channel_group Output channel group index.
 * @param[in] mul_add Serialized multipliers and biases.
 * @return Pointer immediately after the written output.
 */
C_API int16_t *output_transform_fn_int16(otfn_int16_params_t *params,
                                   int16_t *output, int16_t *vDvR,
                                   int32_t output_channel_group,
                                   int32_t *mul_add);

/**
 * @brief Transform int16 output-transform weights and parameters.
 * @param[in] kernel_weights_in Input kernel weights.
 * @param[in] channel_multipliers_in Per-channel floating-point multipliers.
 * @param[in] channel_bias_terms_in Per-channel bias terms.
 * @param[out] kernel_weights_out Reordered output weights.
 * @param[out] mul_add_out Quantized multipliers and biases.
 * @param[in] input_channels Number of input channels.
 * @param[in] output_channels Number of output channels.
 */
C_API void output_transform_fn_int16_kernel_transform(
    const int8_t *kernel_weights_in, const float *channel_multipliers_in,
    const int *channel_bias_terms_in, int8_t *kernel_weights_out,
    int32_t *mul_add_out, int input_channels, int output_channels);

/** Output-transform channel mappings. */
extern int ot_int16_mul_index_used_for_output[];
extern int ot_int16_add_index_used_for_output[];
extern int aggr_ot_int16_input_channel_used_for_output[];

/**
 * @brief Build a quadratic approximation table.
 * @param[out] table Table to populate.
 * @param[in] function Function to approximate.
 * @param[in] input_scaler Scale applied to the input.
 * @param[in] output_scaler Scale applied to the output.
 * @param[in] chunks Number of interpolation chunks.
 * @param[out] max_error Maximum approximation error.
 * @param[out] error Sum-of-squared-error metric.
 */
C_API void quadratic_approximation_generator(
    quadratic_function_table_t *table,
    ACTIVATION_FUNCTION float_function_t function,
    double input_scaler, double output_scaler, int chunks, int *max_error,
    double *error);

/**
 * @brief Return the number of bytes used by a quadratic approximation table.
 * @param[in] table Approximation table.
 * @return Number of serialized table bytes.
 */
C_API uint32_t quadratic_function_table_number_bytes(
    quadratic_function_table_t *table);

/**
 * @brief Return the serialized bytes of a quadratic approximation table.
 * @param[in] table Approximation table.
 * @return Pointer to the serialized table bytes.
 */
C_API uint8_t *quadratic_function_table_bytes(
    quadratic_function_table_t *table);

/**
 * @brief Hyperbolic tangent activation function.
 * @param[in] x Input value.
 */
C_API float approximation_function_tanh(float x);

/**
 * @brief Logistic activation function.
 * @param[in] x Input value.
 */
C_API float approximation_function_logistics(float x);

/**
 * @brief Exponential linear unit activation function.
 * @param[in] x Input value.
 */
C_API float approximation_function_elu(float x);

/**
 * @brief Rectified linear unit activation function.
 * @param[in] x Input value.
 */
C_API float approximation_function_relu(float x);

/**
 * @brief ReLU6 activation function.
 * @param[in] x Input value.
 */
C_API float approximation_function_relu6(float x);

/**
 * @brief Evaluate a quadratic approximation table for an int16 vector.
 * @param[out] outputs Output vector.
 * @param[in] inputs Input vector.
 * @param[in] coeffs Serialized approximation coefficients.
 * @param[in] length Number of elements to process.
 */
C_API void quadratic_interpolation_128(int16_t *outputs, int16_t *inputs,
                                 uint8_t *coeffs, uint32_t length);

/**
 * @brief Add two int16 tensors using a transformed parameter blob.
 * @param[out] output Output tensor.
 * @param[in] input1 First input tensor.
 * @param[in] input2 Second input tensor.
 * @param[in] tensor_length Number of tensor elements.
 * @param[in] blob Transformed parameter blob.
 */
C_API void add_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2,
                      int tensor_length, void *blob);

/**
 * @brief Create the transformed parameter blob for add_int16_tensor().
 * @param[out] output Output blob.
 * @param[in] input1_scaler First input scale.
 * @param[in] input2_scaler Second input scale.
 * @param[in] output_scaler Output scale.
 * @param[out] err_msg Error message buffer.
 * @return Nonzero on success.
 */
C_API int add_int16_tensor_blob(void *output, float input1_scaler,
                                float input2_scaler, float output_scaler,
                                char *err_msg);

/**
 * @brief Dequantize an int16 tensor into a float tensor.
 * @param[out] output Output float tensor.
 * @param[in] input Input int16 tensor.
 * @param[in] tensor_length Number of tensor elements.
 * @param[in] blob Transformed parameter blob.
 */
C_API void dequantize_int16_tensor(float *output, int16_t *input, int tensor_length,
                             void *blob);

/**
 * @brief Create the transformed parameter blob for dequantize_int16_tensor().
 * @param[out] output Output blob.
 * @param[in] input_scaler Input scale.
 * @param[out] err_msg Error message buffer.
 * @return Nonzero on success.
 */
C_API int dequantize_int16_tensor_blob(void *output, float input_scaler,
                                       char *err_msg);

/**
 * @brief Quantize a float tensor into an int16 tensor.
 * @param[out] output Output int16 tensor.
 * @param[in] input Input float tensor.
 * @param[in] tensor_length Number of tensor elements.
 * @param[in] blob Transformed parameter blob.
 */
C_API void quantize_int16_tensor(int16_t *output, float *input, int tensor_length,
                           void *blob);

/**
 * @brief Create the transformed parameter blob for quantize_int16_tensor().
 * @param[out] output Output blob.
 * @param[in] output_scaler Output scale.
 * @return Nonzero on success.
 */
C_API int quantize_int16_tensor_blob(void *output, float output_scaler);

/**
 * @brief Expand an int8 tensor into an int16 tensor.
 * @param[out] out Output int16 tensor.
 * @param[in] in Input int8 tensor.
 * @param[in] length Number of elements to process.
 */
C_API void expand_8_to_16(int16_t *out, int8_t *in, int length);

/**
 * @brief Create the transformed parameter blob for int16 requantization.
 * @param[out] output Output blob.
 * @param[in] input_scaler Input scale.
 * @param[in] output_scaler Output scale.
 * @param[out] err_msg Error message buffer.
 * @return Nonzero on success.
 */
C_API int requantize_int16_tensor_blob(void *output, float input_scaler,
                                       float output_scaler, char *err_msg);

/**
 * @brief Multiply two int16 tensors using a transformed parameter blob.
 * @param[out] output Output tensor.
 * @param[in] input1 First input tensor.
 * @param[in] input2 Second input tensor.
 * @param[in] tensor_length Number of tensor elements.
 * @param[in] blob Transformed parameter blob.
 */
C_API void multiply_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2,
                           int tensor_length, void *blob);

/**
 * @brief Create the transformed parameter blob for multiply_int16_tensor().
 * @param[out] output Output blob.
 * @param[in] input1_scaler First input scale.
 * @param[in] input2_scaler Second input scale.
 * @param[in] output_scaler Output scale.
 * @param[out] err_msg Error message buffer.
 * @return Nonzero on success.
 */
C_API int multiply_int16_tensor_blob(void *output, float input1_scaler,
                                     float input2_scaler, float output_scaler,
                                     char *err_msg);

/**
 * @brief Requantize an int16 tensor using a transformed parameter blob.
 * @param[out] output Output tensor.
 * @param[in] input Input tensor.
 * @param[in] tensor_length Number of tensor elements.
 * @param[in] blob Transformed parameter blob.
 */
C_API void requantize_int16_tensor(int16_t *output, int16_t *input, int tensor_length,
                             void *blob);
