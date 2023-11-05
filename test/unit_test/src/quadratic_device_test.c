#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "quadratic_approximation.h"
#include "quadratic_interpolation.h"

int main(void) {
    double square_error;
    int max_error;
    int chunks = 128;
    float_function_t test_functions[3]  = {approximation_function_tanh,
                                           approximation_function_logistics,
                                           approximation_function_elu};
    float output_scalers[3] = {1.0/32768,
                               1.0/32768,
                               10.0/32768};
    float input_scalers[3] = {8.0/32768,
                              8.0/32768,
                              2.0/32768};
    for(int f = 0; f < 3; f++) {
        quadratic_function_table_t *table = 
            quadratic_approximation_generator(test_functions[f],
                                              input_scalers[f],
                                              output_scalers[f], chunks, &max_error,
                                              &square_error);
        printf("Max error %d sqerror %f\n", max_error, square_error);
        int16_t inputs[655];
        int16_t outputs[655];
        for(int i = 0; i < 655; i++) {
            inputs[i] = i*100-32768;
        }
        uint8_t *bytes = quadratic_function_table_bytes(table);
        quadratic_interpolation_128(outputs, inputs, bytes, 655);
        for(int i = 0; i < 655; i++) {
            float expected = (test_functions[f])(inputs[i] * input_scalers[f]) / output_scalers[f];
            int err = outputs[i] - round(expected);
            if (abs(err) > 1) {
                printf("ERROR\n");
            }
        }
        free(table);
    }
}


