#ifndef _multiply_int16_h_
#define _multiply_int16_h_

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
void multiply_int16_elementwise_constant(int16_t *output, int16_t *input1,
                                         void *blob, int tensor_length);

#endif
