#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include "multiply_int16.h"

extern void multiply_int16_elementwise_constant_asm(int16_t *output, int16_t *input1, void *blob, int tensor_length);

static void multiply_int16_elementwise_constant_ref(int16_t *output, int16_t *input1, void *blob, int tensor_length) {
    int16_t *transformed_input2 = blob;
    for(int i = 0; i < tensor_length; i++) {
        int lower_i = i & 0xf;
        int higher_i = i & ~0xf;
        int transformed_index = higher_i << 1 | lower_i;
        int shift_index = higher_i << 1 | lower_i | 16;
        int16_t *shift = transformed_input2 + tensor_length;
        int mult = input1[i] * transformed_input2[transformed_index];
        mult = (mult + (1<<(shift[i]-1))) >> transformed_input2[shift_index];
        if (mult > 32767) mult = 32767;
        if (mult < -32768) mult = -32768;
        output[i] = mult;
    }
}

void multiply_int16_elementwise_constant(int16_t *output, int16_t *input1, void *transformed_input2, int tensor_length) {
#ifdef NN_USE_REF
    multiply_int16_elementwise_constant_ref(output, input1, transformed_input2, tensor_length);
#else
    multiply_int16_elementwise_constant_asm(output, input1, transformed_input2, tensor_length);
#endif
}


#ifdef LOCAL_MAIN

#include "multiply_int16_transform.h"

#define N 25
int16_t input1[N];
int16_t input2[N];
int8_t transformed_input2[MULTIPLY_INT16_BYTES(N)];
int8_t requantise_blob[REQUANTISE_INT16_BYTES(N)];
int16_t output[N];
int16_t ref_output[N];
int16_t req_output[N];

int main(void) {
    int errors = 0;
    for(int i = 0; i < N; i++) {
        input1[i] = 20000 - 2513 * i;
        input2[i] = 417 * i + 82;
    }
    input2[3] = 2;
    input2[4] = 1;
    input2[5] = -1;
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00413321;
    float scaler2 = 0.000190654;
    float scalero = 0.00318776123;
    for(int i = 0; i < N; i++) {
        float oo = input1[i] * input2[i] * scaler1 * scaler2 / scalero;
        float o = round(oo);
        if (o >  32767) o =  32767;
        if (o < -32768) o = -32768;
        ref_output[i] = o;
        oo = input1[i] * scaler1 / scalero;
        o = round(oo);
        if (o >  32767) o =  32767;
        if (o < -32768) o = -32768;
        req_output[i] = o;
    }
    multiply_int16_transform(transformed_input2,
                             input2,
                             N,
                             scaler1,
                             scaler2,
                             scalero);
    multiply_int16_elementwise_constant(output, input1, transformed_input2, N);

    for(int i = 0; i < N; i++) {
        if (abs(output[i] - ref_output[i]) > 1) {
            printf("%d: %d %d\n", i, output[i], ref_output[i]);
            errors++;
        }
    }
    requantise_int16_transform(requantise_blob,
                               N,
                               scaler1,
                               scalero);
    multiply_int16_elementwise_constant(output, input1, requantise_blob, N);

    for(int i = 0; i < N; i++) {
        if (abs(output[i] - req_output[i]) > 1) {
            printf("%d: %d %d\n", i, output[i], req_output[i]);
            errors++;
        }
    }
    if (errors == 0) {
        printf("PASS\n");
    } else {
        printf("FAIL: %d errors\n", errors);
    }
}
#endif
