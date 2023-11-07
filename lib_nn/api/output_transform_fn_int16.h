#ifndef _output_transform_fn_int16_h_
#define _output_transform_fn_int16_h_

#include <stdint.h>

typedef struct {
    int32_t output_slice_channel_count;
} otfn_int16_params_t;


/** Function that transform a ring buffer accumulator
 * into a vector of 16 bit numbers after multiplying and scaling.
 * In the name of efficiency the inputs shoudl be provided in the following
 * order (K indicates the value that affects output channel K):
 *
 * vD/vR:   0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15.
 *          (each second and third element are swapped)
 *          If only few channels are being used, eg two, they
 *          must be: 0, X, 1, X, X, X, X, X X, X, X, X X, X, X, X.
 *
 * mul_add: m1, m3, m5, m7, m9, m11, m13, m15,
 *          a1, a3, a5, a7, a9, a11, a13, a15,
 *          m0, m2, m4, m6, m8, m10, m12, m14
 *          a0, a2, a4, a6, a8, a10, a12, a14.
 *          If only few channels are being used, eg two, they
 *          must be: m1, X, X, X, X, X, X, X, a1, X, X, X, X, X, X, X,
 *                   m0, X, X, X, X, X, X, X, a0, X, X, X, X, X, X, X.
 *
 * output:  0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
 *          Only the channels used are written, eg, for two channels
 *          0, 1.
 *
 * 
 *
 * \param  vDvR    Pointer to 32 shorts storing 16 upper halfs
 *                 of the ring buffer, and 16 lower halfs in that
 *                 order
 *
 * \param  mul_add Pointer to four vectors of length N/2 integers, the first
 *                 and third vector are multipliers (Q2.30), the second
 *                 and fourth vectors are adders (Q16.16).
 *
 * \param  output  Pointer to the desired place where N values will be
 *                 outputted
 *
 * \param  N       Number of vector elements to process; N <= 16.
 */
extern int16_t *output_transform_fn_int16(otfn_int16_params_t *params,
                                          int16_t *output,
                                          int16_t *vDvR,
                                          int32_t output_channel_group,
                                          int32_t *mul_add);

#endif
