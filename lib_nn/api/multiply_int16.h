#ifndef _multiply_int16_h_
#define _multiply_int16_h_

/**
 * Function that implements a multiplication of two 16-bit tensors where
 * one tensor is constant. The constant tensor must have been converted to 
 * a blob at ompile time using multiply_int16_transform or requantise_1t16_transform
 *
 * @param output         Output tensor
 *
 * @param input          Non constant input tensor
 *
 * @param blob           Transformed constant input tensor
 *
 * @param tensor_length  Number of elements in the tensor (product of all dimensions)
 */
void multiply_int16_elementwise_constant(int16_t *output, int16_t *input1,
                                         void *blob, int tensor_length);

#endif