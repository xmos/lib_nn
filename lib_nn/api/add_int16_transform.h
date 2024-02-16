#ifndef _add_int16_transform_h_
#define _add_int16_transform_h_

#include "nn_api.h"
#include <stdint.h>

/**
 * Function that performs the compile time transformation of an int16 addition
 * between two tensors.
 * 
 * this function should be
 * called at compile-time, and at run-time the output of this function shall be passed
 * as the second input tensor of ``add_int16_tensor``
 * 
 * @param output            Output of the function; a blob of
 *                          ``ADD_INT16_TENSOR_BYTES()`` bytes.
 *                          Must be word-aligned.
 *
 * @param input1_scaler     Quantisation scaler for input1
 *                          Negate this to make add compute -A+B
 *
 * @param input2_scaler     Quantisation scaler for input2
 *                          Negate this to make add compute A-B
 *
 * @param output_scaler     Quantisation scaler for output
 *
 * @returns 1 on success, 0 on fail (fallback required)
 */
C_API int add_int16_tensor_blob(void *output,
                          float input1_scaler,
                          float input2_scaler,
                          float output_scaler);

/**
 * Macro that calculates the number of int16_t that should be allocated to
 * store the output of ``add_int16_tensor_blob()``
 */
#define ADD_INT16_TENSOR_BYTES()  (2 * 16 * sizeof(int16_t))


#endif
