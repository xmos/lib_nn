#include <stdio.h>
#include <stdlib.h>
#include "quadratic_approximation.h"

int main(void) {
    double square_error;
    int max_error;
    for(int chunks = 128; chunks < 129; chunks *= 2) {
        quadratic_function_table_t *output = 
            quadratic_approximation_generator(approximation_function_tanh,
                                             8.0/32768, 1.0/32768, chunks, &max_error,
                                             &square_error);
        if (chunks == 129) {
            uint8_t *bytes = quadratic_function_table_bytes(output);
            int number = quadratic_function_table_number_bytes(output);
            printf("uint8_t coeffs[] = {\n");
            for(int i = 0; i < number; i++) {
                printf("    0x%02x,\n", bytes[i]);
            }
            printf("};\n");
        }
        free(output);
    }
}


