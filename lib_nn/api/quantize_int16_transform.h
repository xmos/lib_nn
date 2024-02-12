#ifndef _quantize_int16_transform_h_
#define _quantize_int16_transform_h_

#include "nn_api.h"
#include <stdint.h>

/**
 * Function that performs the compile time transformation of an int16 addition
 * between two tensors.
 * 
 * this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``quantize_int16_tensor``
 * 
 * @param output            Output of the function; a blob of
 *                          ``QUANTIZE_INT16_TENSOR_BYTES()`` bytes.
 *                          Must be word-aligned.
 *
 * @param output_scaler      Quantisation scaler for the output
 *
 * @returns 1 on success, 0 on fail (fallback required)
 */
C_API int quantize_int16_tensor_blob(void *output,
                                 float output_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``quantize_int16_tensor_blob()``
 */
#define QUANTIZE_INT16_TENSOR_BYTES()  (1 * sizeof(float))


#endif
