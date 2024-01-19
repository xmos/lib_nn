#ifndef _multiply_int16_h_
#define _multiply_int16_h_

#if 0
DEPRECATED

/**
 * Function that implements a multiplication of two 16-bit tensors where
 * one tensor is constant. The constant tensor must have been converted to 
 * a blob at ompile time using multiply_int16_transform or requantise_1t16_transform
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
void multiply_int16_constant(int16_t *output, int16_t *input1,
                                         void *blob, int tensor_length);

#endif

/**
 * Function that implements a multiplication of two 16-bit tensors where
 * both tensors are variable. The blob must have been created by a call to 
 * ``multiply_int16_blob()``
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
 * created at compile time using ``requantise_int16_blob()``
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
void quantise_int16_tensor(int16_t *output, int16_t *input1, void *blob, int tensor_length);

#endif
