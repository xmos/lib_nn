#ifndef _multiply_int16_transform_h_
#define _multiply_int16_transform_h_

#include "nn_api.h"
#include <stdint.h>


/**
 * Function that performs the compile time transformation of an int16 to int16
 * re-quantisation.
 * 
 * For an int16 re-quantisation, this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``requantise_int16_tensor``
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
C_API int requantize_int16_tensor_blob(void *output,
                                 float input_scaler,
                                 float output_scaler);
/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``quantise_int16_tensor_blob()``
 */
#define REQUANTIZE_INT16_TENSOR_BYTES()  (16 * sizeof(int16_t))


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
C_API int multiply_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``multiply_int16_tensor_blob()``
 * TODO: this could be zero and stored in the pointer...
 */
#define MULTIPLY_INT16_TENSOR_BYTES()  (2 * sizeof(int16_t))


#endif
