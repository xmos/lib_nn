#ifndef _multiply_int16_transform_h_
#define _multiply_int16_transform_h_

#include <stdint.h>

/**
 * Function that performs the compile time transformation of an int16 multiplication
 * between a tensor and a constant tensor.
 * 
 * For an int16 elementwise multiplication with a constant tensor, this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``multiply_int16``
 * 
 * @param output            Output of the function; a blob that is 
 *                          MULTIPLY_INT16_BYTES(tensor_length) bytes
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
 */
void multiply_int16_transform(void *output,
                              int16_t *constant_multiplier_tensor,
                              int tensor_length,
                              float input1_scaler,
                              float input2_scaler,
                              float output_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of multiply_int16_transform
 */
#define MULTIPLY_INT16_BYTES(tensor_length)  (((2 * tensor_length + 16) & ~15) * sizeof(int16_t))

/**
 * Function that performs the compile time transformation of an int16 to int16
 * re-quantisation.
 * 
 * For an int16 requantisation, this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``multiply_int16``
 * 
 * @param output            Output of the function; a blob of
 *                          REQUANTISE_INT16_BYTES(tensor_length) bytes.
 *                          Must be word-aligned.
 *
 * @param tensor_length     Number of elements in the input tensor
 *
 * @param input1_scaler     Quantisation scaler for input
 *
 * @param output_scaler     Quantisation scaler for output
 *
 */
void requantise_int16_transform(void *output,
                                int tensor_length,
                                float input_scaler,
                                float output_scaler);
/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of requantise_int16_transform
 */
#define REQUANTISE_INT16_BYTES(tensor_length)  (((2 * tensor_length + 16) & ~15) * sizeof(int16_t))


#endif
