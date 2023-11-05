#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include "quadratic_approximation.h"
#include "quadratic_interpolation.h"

struct quadratic_function_table {
    struct {          // The order matters - this is how the assembly code expects them
        int32_t c;
        int8_t a;
        int8_t padding;
        int16_t b;
    } coefficients[QUADRATIC_APPROXIMATION_MAX_CHUNKS];
    int data_bytes;
};

/*
 * Algorithm for interpolation of activation functions
 * 
 * We chop the domain up into CHUNKS equal chunks; lets assume CHUNKS=128, the domain
 * is 16-bit values between -32768 and 32767, so each chunk will contain 512 elements
 * of the domain. so the elements of chunk 62 are elements -1024..-513.
 *
 * Inside each chunk we are going to make a quadratic interpolation function of the form:
 *
 *   a * x^2 + b * x^1 + c
 *
 * Here x is a normalised x value that is zero in the centre of each chunk. The
 * coefficients a, b, and c are coefficients that are calculated for this chunk.
 * In order to calcuate, for example, tanh(-1023) we would first find out which
 * chunk it is (-1023 + 32768) / 512 = 62; then within the chunk calculate
 * x (-1023 - -1024 - 256 = -255), lookup a[62], b[62], and c[62] and calculate tanh
 * using the formula above.
 *
 * The trick is to use integer arithmetic. In order to get a good trade-off between
 * accuracy, implementation speed, and storage we quantise a, b, and c into A, B, and C
 * as follows:
 * 
 *   A = a * N * N
 *   B = b * N
 *   C = c * N * N
 *
 * Where N = 256. Given that a is typically very small (1e-5), A easily fits in an
 * 16-bit value. b is typically no more than 6 or so, and hence B also fits in a 16-bit
 * value. c is in the range [-32768..32767], and is shifted up to 16 bits to keep a
 * rounding bit and to facilitate fast loading and processing.
 * 
 * Substitution of the above values for a, b, and c yields
 *
 *   A / N / N * x^2 + B / N * x^1 + c / N / N
 *
 * Factoring out / N / N yields
 *
 *   ( A * x^2 + B * N * x^1 + c) / N / N
 *
 * We can now rewrite the centre:
 *
 *   (( A * x + B * N ) * x + c) / N / N
 *
 * And we can write these with multiply accumulate functions
 *
 *   macc(macc(A, x, B*N), x, C) / (N*N)
 * 
 * where macc(k,l,m) = k * l + m; which is a single instruction where k and l are
 * 32-bits, and m is 64 bits.
 * Note that the inner maxx(A, x, B*N) will guaranteed fit in 32-bits given the values
 * involved, and therefore can be used as a 32-bit value for the outer one. B*N has to be
 * sign extended, but for monotinous functions B is always positive. C has to be sign
 * extended to 64 bits.
 * 
 * Note that given that N is 2^8, division by (N*N) is equivalent to a shift right
 * of 16 bits. Rounding is achieved by adding 32768 to C prior.
 *
 * Outline assembly code for the calculation is as follows (CPU only):
 *
 *    ld16s X[v]                 
 *    ld16s CH[v]              
 *    ldd   AB, C, [CH]       
 *    shr   B, -N; sext A, 16 
 *    ldc zero, 0
 *    macc  zero, B, A, X  
 *    shl signC, C, -32    
 *    macc  signC, C, B, X 
 *    lsats zero, C, 0     
 *    shr   C, 16         
 *    st16  C, X[v]
 *
 * In order to calculate A, B, and C for each chunk we use least-squares with a
 * sweeping algorithm to invert the inevitable 3x3 matrix. At present the algorithm
 * asserts if the matrix cannot be inverted; there may be cases where this is
 * undesirable and other action should be taken.
 *
 */


static int clamp(double x) {
    if (x > 32767) return 32767;
    if (x < -32768) return 32768;
    return x;
}

static int clamp8(double x) {
    if (x > 127) return 127;
    if (x < -128) return -128;
    return x;
}

static int clamp32(double x) {
    if (x > 0x7fffffff) return 0x7fffffff;
    if (x < -2147483648.0) return 0x80000000;
    return x;
}

float approximation_function_tanh(float x) {
    return tanh(x);
}

float approximation_function_logistics(float x) {
    return 1.0 /(1.0 + exp(-x));
}

float approximation_function_elu(float x) {
    return x >= 0 ? x : expm1(x);
}

quadratic_function_table_t *quadratic_approximation_generator(
    float_function_t av, double input_scaler,
    double output_scaler, int chunks,
    int *max_error,
    double *error) {
    quadratic_function_table_t *output = calloc(1, sizeof(quadratic_function_table_t));

    assert(chunks <= QUADRATIC_APPROXIMATION_MAX_CHUNKS);
    const int datapoints = 65536 / chunks;
    const int degree = 3;
    double A[datapoints][degree] ;
    double B[datapoints] ;
    double ATA[degree][degree] ;
    double ATB[degree] ;
    int zeropoint = 32768;
    int avg2error_i = 0;
    int max_error_i = 0;
    output->data_bytes = 2 * (chunks) * (degree+1);
    int output_index = 0;
    for(int mid = datapoints/2; mid <= 65536; mid += datapoints) {
        int start = mid - datapoints / 2;
        int16_t inputs_16bit[datapoints];
        int16_t outputs_16bit[datapoints];
        for(int i = 0; i < datapoints; i++) {
            int input_val = i - datapoints / 2;
            A[i][0] = 1;
            for(int d = 1; d < degree; d++) {
                A[i][d] = A[i][d-1] * input_val;
            }
            int real_input_val = i + start - zeropoint;
            inputs_16bit[i] = real_input_val;
            float f_real_input_val = real_input_val * input_scaler;
            B[i] = av(f_real_input_val) / output_scaler;
            if( start + i == 0x3fe3) {
//                printf("Ch %d Tanh %f -> %f\n", chunks, f_real_input_val, B[i]);
            }
        }
        for(int i=0 ; i<degree ; i++ ) {
            for(int j=0 ; j<degree ; j++ ) {
                ATA[i][j] = 0.0 ;
                for(int k=0 ; k<datapoints ; k++ ) {
                    ATA[i][j] += A[k][i] * A[k][j] ;
                }
            }
        }
        for(int i=0 ; i<degree ; i++ ) {
            ATB[i] = 0.0 ;
            for(int k=0 ; k<datapoints ; k++ ) {
                ATB[i] += A[k][i] * B[k] ;
            }
        }
        for(int i=0 ; i<degree ; i++ ) {
            assert(ATA[i][i] != 0.0);
            for(int j=i+1 ; j<degree ; j++ ) {
                if( ATA[j][i] != 0 ) {
                    double fac = ATA[j][i] / ATA[i][i] ;
                    for(int k=i ; k<degree ; k++ ) {
                        ATA[j][k] -= fac * ATA[i][k] ;
                    }
                    ATB[j] -= fac * ATB[i] ;
                }
            }
        }
        for(int i=degree-1 ; i>=0 ; i-- ) {
            assert(ATA[i][i] != 0.0);
            for(int j=0 ; j<i ; j++ ) {
                if( ATA[j][i] != 0 ) {
                    double fac = ATA[j][i] / ATA[i][i] ;
                    for(int k=0 ; k<=i ; k++ ) {
                        ATA[j][k] -= fac * ATA[i][k] ;
                    }
                    ATB[j] -= fac * ATB[i] ;
                }
            }
        }
        for(int i=0 ; i<degree ; i++ ) {
            ATB[i] /= ATA[i][i] ;
        }

        int i_scale_factor = 256;
        ATB[0] = round(ATB[0]*i_scale_factor*i_scale_factor)+32768.0;
        ATB[1] = round(ATB[1]*i_scale_factor);
        ATB[2] = round(ATB[2]*i_scale_factor*i_scale_factor);
        if (ATB[0] >= (1LL<<31)+0x8000  || ATB[0] < -(1LL<<31)-0x8000) {
            printf("Warning: Constant constant -2^31 <= %f < 2^31 out of range\n", ATB[0]);
        }
        if (ATB[1] >= (1<<15)  || ATB[1] < -0.1) {
            printf("Warning: Linear constant 0 <= %f < 32768 out of range\n", ATB[1]);
            if (ATB[1] < 0) ATB[1] = 0;
        }
        if (ATB[2] > 127.5 || ATB[2] < -128.5) {
            printf("Warning: Quadratic constant -127 < %f < 128 out of range\n", ATB[2]);
        }
        output->coefficients[output_index].c = clamp32(ATB[0]);
        output->coefficients[output_index].b = clamp(ATB[1]);
        output->coefficients[output_index].a = clamp8(ATB[2]);
        output->coefficients[output_index].padding = 0;

        quadratic_interpolation_128(outputs_16bit, inputs_16bit,
                                    output, datapoints);
        for(int j = 0 ; j < datapoints; j++) {
            int error_i = round(B[j]) - outputs_16bit[j];
//            printf("XX %04x %04x %f\n", inputs_16bit[j], outputs_16bit[j], round(B[j]));
            if( abs(error_i) > 1 && chunks == 128) {
                printf("Ch %d start %d val %f %08x %f %d\n", chunks, start, (start + j-32768) * input_scaler, (int)round(B[j]), B[j], error_i);
            }
            if (abs(error_i) > max_error_i) {
                max_error_i = abs(error_i);
            }
            avg2error_i += error_i * error_i;
        }
        output_index++;
    }
    *error = sqrt(avg2error_i / 65536.0);
    *max_error = max_error_i;
    return output;
}

uint32_t quadratic_function_table_number_bytes(quadratic_function_table_t *x) {
    return x->data_bytes;
}

uint8_t *quadratic_function_table_bytes(quadratic_function_table_t *x) {
    return (uint8_t *)&x->coefficients;
}
