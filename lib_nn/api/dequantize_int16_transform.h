#ifndef _dequantize_int16_transform_h_
#define _dequantize_int16_transform_h_

#include "nn_api.h"
#include <stdint.h>

/**
 * Function that performs the compile time transformation of an int16 addition
 * between two tensors.
 * 
 * this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``dequantize_int16_tensor``
 * 
 * @param output            Output of the function; a blob of
 *                          ``DEQUANTIZE_INT16_TENSOR_BYTES()`` bytes.
 *                          Must be word-aligned.
 *
 * @param input_scaler      Quantisation scaler for the input
 *
 * @returns 1 on success, 0 on fail (fallback required)
 */
C_API int dequantize_int16_tensor_blob(void *output,
                                 float input_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``dequantize_int16_tensor_blob()``
 */
#define DEQUANTIZE_INT16_TENSOR_BYTES()  (2 * sizeof(float))


#endif
