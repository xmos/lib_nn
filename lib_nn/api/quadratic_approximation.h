#ifndef _quadratic_approximation_h_
#define _quadratic_approximation_h_

#include <stdint.h>

#define QUADRATIC_APPROXIMATION_MAX_CHUNKS   2048

/** Type that stores an approximation table.
 * Must be stored 64-bit aligned when presented to assembly code.
 * Use:
 *    * ``quadratic_approximation_generator()`` to create a table
 *    * ``quadratic_function_table_number_bytes()`` to query the size
 *    * ``quadratic_function_table_bytes()`` to obtain a pointer to the table
 */
typedef struct quadratic_function_table quadratic_function_table_t;

/* Function pointer type - any float -> float
 */
typedef float (*float_function_t)(float x);

/** Function that builds a quadratic approximation table
 * The function passed in must be monotinuous
 * Any number of chunks will work but the assembly implementiaton assumes 128.
 * The function returns the maximum error (an int, ought to be 1), a table
 * (through a pointer), and the sqrt of the sum of squared errors as a goodness metric.
 *
 * \param av            function to be interpolated
 * \param input_scaler  scale that is applied to the input, eg 8.0/32768.0
 * \param outptu_scaler scale that is applied to the output, eg 32768.0
 * \param chunks        number of interpolations. Set to 128.
 * \param [out] output  table with output values
 * \param error         sqrt of sum of squared errors.
 */
extern int quadratic_approximation_generator(float_function_t av,
                                             double input_scaler,
                                             double output_scaler,
                                             int chunks,
                                             quadratic_function_table_t *output,
                                             double *error);

/** Function that returns the number of bytes in an approximation table
 *
 * \param x   the table
 * \returns   The number of bytes in the table
 */
extern uint32_t quadratic_function_table_number_bytes(quadratic_function_table_t *x);


/** Function that returns a pointer to the bytes in an approximation table
 *
 * \param x   the table
 * \returns   Pointer to the bytes in the table
 */
extern uint8_t *quadratic_function_table_bytes(quadratic_function_table_t *x);

/** Example functions that can be passed in 
 */
extern float approximation_function_tanh(float x);
extern float approximation_function_logistics(float x);
extern float approximation_function_elu(float x);

#endif
