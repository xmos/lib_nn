#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "dequantize_int16.h"

// Element dequantisaition
// Convert an int to a float without a cast, by adding 0x40008000
// that makes it a number in the range 2 + 1/64 + intvalue / 2^22
// Then subtract 0x40008000 away (as a float value), and multiply by 2^22 and a scalar
// These two values are precomputed in a blob.
// The clever bit is that we avoid normalisation and let the float addition take care of that.

extern void dequantize_int16_tensor_asm(float *output, int16_t *input, int tensor_length, void *blob);

void dequantize_int16_tensor_ref(float *output, int16_t *input, int tensor_length, void *blob) {
    for(int i = 0; i < tensor_length; i++) {
        float a;
        *(int *) & a = input[i] + 0x40008000;
        output[i] = (a +  (double) ((float *) blob)[1]) * (double) ((float *) blob)[0] ;
    }
}

void dequantize_int16_tensor(float *output, int16_t *input1, int tensor_length, void *blob) {
#ifdef NN_USE_REF
    dequantize_int16_tensor_ref(output, input1, tensor_length, blob);
#else
    dequantize_int16_tensor_asm(output, input1, tensor_length, blob);
#endif
}

#ifdef TEST_MAIN
#include "dequantize_int16_transform.h"

int test(int16_t input, float scalar) {
    int errors = 0;
    float output;
    int blob[2];
    char error[80];
    int r = dequantize_int16_tensor_blob(blob, scalar, error);
    if (r == 0) {
        printf("ERROR: %s\n", error);
    }
    dequantize_int16_tensor_ref(&output, &input, 1, blob);
    float expect = input * scalar;
    if (expect != output) {
        printf("%d * %f -> %f != %f err %f\n", input, scalar, output, expect, output-expect);
        errors++;
    }
    dequantize_int16_tensor_asm(&output, &input, 1, blob);
    expect = input * scalar;
    if (expect != output) {
        printf("%d * %f -> %f != %f err %f\n", input, scalar, output, expect, output-expect);
        errors++;
    }
    return errors;
}

int testscalar(int16_t input) {
    int errors = 0;
    errors += test(input, 0.00001);
    errors += test(input, 0.001);
    errors += test(input, 0.1);
    errors += test(input, 10.0);
    return errors;
}

int main(void) {
    int errors = 0;
    errors += testscalar(-32768);
    errors += testscalar( 32767);
    errors += testscalar(-3);
    errors += testscalar( 3);
    printf("%d errors\n", errors);
    return 0;
}

#endif
