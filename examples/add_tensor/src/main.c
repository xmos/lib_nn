// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <assert.h>

#include "nn_api.h"
#include "nn_layers.h"

#define SIZE 12

static 
void print_array(const char *name, int16_t *array, unsigned length) {
    printf("%s: ", name);
    for (unsigned i = 0; i < length; i++) {
        printf("%d,\t", array[i]);
    }
    printf("\n");
}

int main() {
    WORD_ALIGNED int16_t input1[SIZE] = {
        -100,200,300,400,-500,800,100,-50,-25,1000,1100,1200
    };
    WORD_ALIGNED int16_t input2[SIZE] = {
        100,200,300,400,500,600,700,800,900,1000,1100,1200
    };
    WORD_ALIGNED int16_t output[SIZE];
    
    int8_t blob[ADD_INT16_TENSOR_BYTES()];
    char err_msg[ERR_MSG_DESCRIPTOR_FAIL_BYTES()];

    // Compute fixed-point multipliers to rescale each input into the output scale
    // Here inputs and output are in the same scale, result is performed without rescaling.
    int ret = add_int16_tensor_blob(blob, 1, 1, 1, err_msg);
    assert(ret && "Failed to create add_int16 tensor blob");

    // perform the addition
    add_int16_tensor(output, input1, input2, SIZE, blob);

    // print the results
    print_array("Input1", input1, SIZE);
    print_array("Input2", input2, SIZE);
    print_array("Output", output, SIZE);
    return 0;
}
