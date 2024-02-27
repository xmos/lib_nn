#ifndef _quadratic_approximation_h_
#define _quadratic_approximation_h_

#ifdef __xcore__
#define ACTIVATION_FUNCTION __attribute__(( fptrgroup("activation_functions") ))
#else
#define ACTIVATION_FUNCTION /**/
#endif

#include "nn_api.h"
#include <stdint.h>

#define QUADRATIC_APPROXIMATION_MAX_CHUNKS   2048

/** Type that stores an approximation table.
 * Must be stored 64-bit aligned when presented to assembly code.
 * Use:
 *    * ``quadratic_function_table_number_bytes()`` to query the size
 *    * ``quadratic_function_table_bytes()`` to obtain a pointer to the table
 */
struct quadratic_function_table {
    struct {          // The order matters - this is how the assembly code expects them
        int32_t c;
        int8_t a;
        int8_t padding;
        int16_t b;
    } coefficients[QUADRATIC_APPROXIMATION_MAX_CHUNKS];
    int data_bytes;
};

typedef struct quadratic_function_table quadratic_function_table_t;

/* Function pointer type - any float -> float
 */
typedef float (*float_function_t)(float x);

/** Function that builds a quadratic approximation table
 * The function passed in must be monotinuous
 * Any number of chunks will work but the assembly implementiaton assumes 128.
 * The function returns a table pointer, and through two arguments the max error,
 * and the sqrt of the sum of squared errors as a goodness metric.
 *
 * \param table         interpolation table to be filled in
 * \param av            function to be interpolated
 * \param input_scaler  scale that is applied to the input, eg 8.0/32768.0
 * \param outptu_scaler scale that is applied to the output, eg 32768.0
 * \param chunks        number of interpolations. Set to 128.
 * \param max_error     maximum error, ought to be 1
 * \param error         sqrt of sum of squared errors.
 */
C_API void quadratic_approximation_generator(
    quadratic_function_table_t *table,
    ACTIVATION_FUNCTION float_function_t av,
    double input_scaler,
    double output_scaler,
    int chunks,
    int *max_error,
    double *error);

/** Function that returns the number of bytes in an approximation table
 *
 * \param x   the table
 * \returns   The number of bytes in the table
 */
C_API uint32_t quadratic_function_table_number_bytes(quadratic_function_table_t *x);


/** Function that returns a pointer to the bytes in an approximation table
 *
 * \param x   the table
 * \returns   Pointer to the bytes in the table
 */
C_API uint8_t *quadratic_function_table_bytes(quadratic_function_table_t *x);

/** Example functions that can be passed in 
 */
C_API float approximation_function_tanh(float x);
C_API float approximation_function_logistics(float x);
C_API float approximation_function_elu(float x);
C_API float approximation_function_relu(float x);

#endif
