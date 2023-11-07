#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "expand_8_to_16.h"

int8_t inputs[64];
int16_t outputs[72];
    
int test_expand_8_to_16() {
    int errors = 0;
    for(int i = 0; i < 64; i++) {
        inputs[i] = i*i;
    }
    for(int j = 0; j < 64; j++) {
        for(int i = 0; i < 72; i++) {
            outputs[i] = i^0xFFFF;
        }
        expand_8_to_16(outputs+4, inputs, j);
        for(int i = 0; i < 72; i++) {
            if (outputs[i] != (int16_t)(i^0xFFFF) && (i < 4 || i >= 68)) {
                printf("Guard overwritten %d %04x %04x\n", i, outputs[i], i^0xFFFF);
                errors++;
            }
        }
        for(int i = 0; i < j; i++) {
            if (outputs[i+4] != inputs[i]) {
                printf("Bad value %d %04x %04x\n", i, outputs[i+4], inputs[i]);
                errors++;
            }
        }
    }
    return errors;
}

int main(void) {
    int errors = 0;
    errors += test_expand_8_to_16();
    return errors;
}
