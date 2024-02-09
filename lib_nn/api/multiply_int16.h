#ifndef _multiply_int16_h_
#define _multiply_int16_h_

/**
 * Function that implements a multiplication of two 16-bit tensors where
 * both tensors are variable. The blob must have been created by a call to 
 * ``multiply_int16_tensor_blob()``
 *
 * @param output         Output tensor
 *                       Must be word-aligned
 *
 * @param input          Non constant input tensor
 *                       Must be word-aligned
 *
 * @param blob           Transformed constant input tensor
 *                       Must be word-aligned
 *
 * @param tensor_length  Number of elements in the tensor (product of all dimensions)
 *                       There are no constraints on this number.
 */
void multiply_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2,
                           int tensor_length, void *blob);

/**
 * Function that implements a requantifies a 16-bit tensor. The blob must have been
 * created at compile time using ``requantise_int16_tensor_blob()``
 *
 * @param output         Output tensor
 *                       Must be word-aligned
 *
 * @param input          Non constant input tensor
 *                       Must be word-aligned
 *
 * @param tensor_length  Number of elements in the tensor (product of all dimensions)
 *                       There are no constraints on this number.
 *
 * @param blob           Transformed constant input tensor
 *                       Must be word-aligned
 */
void requantize_int16_tensor(int16_t *output, int16_t *input1, int tensor_length, void *blob);

#endif
