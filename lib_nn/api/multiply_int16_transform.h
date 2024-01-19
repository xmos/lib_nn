#ifndef _multiply_int16_transform_h_
#define _multiply_int16_transform_h_

#include <stdint.h>

#if 0

DEPRECATED

/**
 * Function that performs the compile time transformation of an int16 multiplication
 * between a tensor and a constant tensor.
 * 
 * For an int16 elementwise multiplication with a constant tensor, this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``multiply_int16_constant``
 * 
 * @param output            Output of the function; a blob that is 
 *                          ``MULTIPLY_INT16_CONSTANT_BYTES(tensor_length)`` bytes
 *                          long. Must be word-aligned.
 *
 * @param constant_multiplier_tensor   The constant input tensor
 *
 * @param tensor_length     Number of elements in the input tensor
 *
 * @param input1_scaler     Quantisation scaler for input1
 *
 * @param input2_scaler     Quantisation scaler for input2
 *
 * @param output_scaler     Quantisation scaler for output
 *
 * @returns 1 on success, 0 on fail (fallback required)
 */
int multiply_int16_constant_blob(void *output,
                                  int16_t *constant_multiplier_tensor,
                                  int tensor_length,
                                  float input1_scaler,
                                  float input2_scaler,
                                  float output_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of multiply_int16_transform
 */
#define MULTIPLY_INT16_CONSTANT_BYTES(tensor_length)  (((2 * tensor_length + 16) & ~15) * sizeof(int16_t) + sizeof(int32_t))
#endif


/**
 * Function that performs the compile time transformation of an int16 to int16
 * re-quantisation.
 * 
 * For an int16 quantisation, this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``multiply_int16_constant``
 * 
 * @param output            Output of the function; a blob of
 *                          ``REQUANTISE_INT16_BYTES()`` bytes.
 *                          Must be word-aligned.
 *
 * @param input1_scaler     Quantisation scaler for input
 *
 * @param output_scaler     Quantisation scaler for output
 *
 * @returns 1 on success, 0 on fail (fallback required)
 */
int quantise_int16_blob(void *output,
                           float input_scaler,
                           float output_scaler);
/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``quantise_int16_blob()``
 */
#define QUANTISE_INT16_BYTES()  (16 * sizeof(int16_t))


/**
 * Function that performs the compile time transformation of an int16 multiplication
 * between two tensors.
 * 
 * this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``multiply_int16_tensor``
 * 
 * @param output            Output of the function; a blob of
 *                          ``MULTIPLY_INT16_TENSOR_BYTES()`` bytes.
 *                          Must be word-aligned.
 *
 * @param input1_scaler     Quantisation scaler for input1
 *
 * @param input2_scaler     Quantisation scaler for input2
 *
 * @param output_scaler     Quantisation scaler for output
 *
 * @returns 1 on success, 0 on fail (fallback required)
 */
int multiply_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``multiply_int16_tensor_blob()``
 */
#define MULTIPLY_INT16_TENSOR_BYTES()  (16 * sizeof(int16_t))


#endif
