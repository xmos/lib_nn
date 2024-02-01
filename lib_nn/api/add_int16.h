#ifndef _add_int16_h_
#define _add_int16_h_

/**
 * Function that implements a addition of two 16-bit tensors.
 * The blob must have been created by a call to 
 * ``add_int16_tensor_blob()``
 *
 * @param output         Output tensor
 *                       Must be word-aligned
 *
 * @param input1         Input tensor operand 1
 *                       Must be word-aligned
 *
 * @param input1         Input tensor operand 2
 *                       Must be word-aligned
 *
 * @param blob           Transformed constant input tensor
 *                       Must be word-aligned
 *
 * @param tensor_length  Number of elements in the tensor (product of all dimensions)
 *                       There are no constraints on this number.
 */
void add_int16_tensor(int16_t *output, int16_t *input1, int16_t *input2,
                      int tensor_length, void *blob);

#endif
