#include <assert.h>
#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include "add_int16_transform.h"

#define SHIFT  14

int add_int16_tensor_blob(void *output,
                               float input1_scaler,
                               float input2_scaler,
                               float output_scaler) {
    int tensor_length = 16;
    int16_t *output_tensor = (int16_t *) output;
    float combined_scaler1 = input1_scaler / output_scaler;
    float combined_scaler2 = input2_scaler / output_scaler;
    assert(combined_scaler1 > 0);
    assert(combined_scaler2 > 0);
    int mult1 = round(combined_scaler1 * (1 << SHIFT));
    int mult2 = round(combined_scaler2 * (1 << SHIFT));
    if (mult1 > 32767 || mult2 > 32767) {
        return 0;
    }
    for(int i = 0; i < tensor_length; i++) {
        output_tensor[i                ] = mult1;
        output_tensor[i + tensor_length] = mult2;
    }
    return 1;
}
