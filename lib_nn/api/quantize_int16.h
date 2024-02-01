#ifndef _quantize_int16_h_
#define _quantize_int16_h_

/**
 * Function that implements quantization of a 16-bit tensor to a 32-bit tensor.
 * The blob must have been created by a call to ``quantize_int16_blob()``
 *
 * @param output         Output tensor
 *
 * @param input          Input tensor
 *                       Must be word-aligned
 *
 * @param blob           Transformed constant input tensor
 *                       Must be word-aligned
 *
 * @param tensor_length  Number of elements in the tensor (product of all dimensions)
 *                       There are no constraints on this number.
 */
void quantize_int16_tensor(int16_t *output, float *input,
                             int tensor_length, void *blob);

#endif
