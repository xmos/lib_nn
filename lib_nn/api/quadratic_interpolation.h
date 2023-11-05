#ifndef _quadratic_interpolation_h_
#define _quadratic_interpolation_h_

#include <stdint.h>
#include "quadration_approximation.h"

/** Function that performs a quadratic interpolation on a vector of inputs given a table
 * of coefficients
 * 
 * \param    outputs       Output vector, 16-bit signed integers
 * \param    inputs        Output vector, 16-bit signed integers
 * \param    coefficients  table of coefficients produced with
 *                         ``quadratic_approximation_generator()``
 * \param    N             Number of 16-bit elements in the vector
 */
extern void quadratic_interpolation_128(int16_t *outputs, int16_t *inputs,
                                        quadratic_function_table_t coeffs[128],
                                        uint32_t N);

#endif
