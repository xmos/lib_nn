#ifndef _dequantize_int16_h_
#define _dequantize_int16_h_

/**
 * Function that implements dequantization of a 16-bit tensor to a 32-bit tensor.
 * The blob must have been created by a call to ``dequantize_int16_tensor_blob()``
 *
 * @param output         Output tensor
 *                       Must be word-aligned
 *
 * @param input          Input tensor
 *
 * @param blob           Transformed constant input tensor
 *                       Must be word-aligned
 *
 * @param tensor_length  Number of elements in the tensor (product of all dimensions)
 *                       There are no constraints on this number.
 */
void dequantize_int16_tensor(float *output, int16_t *input,
                             int tensor_length, void *blob);

#endif
