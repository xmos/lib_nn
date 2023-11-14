#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "output_transform_fn_int16.h"
#include "output_transform_fn_int16_kernel_transform.h"

int16_t vDvR[32] = {
};

int16_t vDvR_input[32] = {
    1,3,2,4,5,7,6,8,
    -8,-6,-7,-5,-4,-2,-3,-1,
    9,10,11,12,13,14,15,16,
    -17,-18,-19,-20,-21,-22,-23,-24
};

int32_t mul_add[32];

int32_t multipliers[16] = {
    0x4000000, 0x4000000,0x4000000, 0x4000000,0x4000000, 0x4000000,0x4000000, 0x4000000,
    0x0000000, 0x4000000,0x4000000, 0x4000000,0x4000000, 0x4000000,0x4000000, 0x4000000
};

int32_t adders[16] = {
    0x1000, 0, 0, 0,  0, 0, 0, 0,
    0, 0, 0, 0,  0, 0, 0, 0,
};

int16_t output[40];


int test_output_transform_fn_int16(void) {
    int errors = 0;
    for(int j = 1; j <= 16; j+=1) {
        otfn_int16_params_t otfn_params = {j};
        memset(vDvR,    0,             j*4);
        memcpy(vDvR,    vDvR_input,    ((j+3)&~3)*2);
        memcpy(vDvR+16, vDvR_input+16, ((j+3)&~3)*2);
        memset(mul_add,    0,          32*4);
        for(int i = 0; i < 2*16; i+=16) {
            memcpy(&mul_add[i],      &multipliers[i/2], 16*2);
            memcpy(&mul_add[i+16/2], &adders[i/2],      16*2);
        }
        for(int i = 0; i < 40; i+=1) {
            output[i] = i*i+13;
        }
        output_transform_fn_int16(&otfn_params, output+4, vDvR, 0, mul_add);
        for(int i = 0; i < 40; i+=1) {
            if (output[i] != i*i+13 && (i < 4 || i >= 4+j)) {
                errors++;
                printf("Padding died %08x should have been %08x\n", output[i], i*i+13);
            }
        }
        for(int i = 0; i < j; i++) {
//            printf("%08lx, ", accs[i]);
        }
//        printf("\n");
        for(int i = 0; i < j; i++) {
            printf("%04x, ", output[i+4] & 0xffff);
        }
        printf("\n");
    }
    return errors;
}

int8_t kernel_weights_in[16*8];
int8_t kernel_weights_out[16*8];
int16_t vDvR[32];
int16_t vDvRoutput[16];
int32_t expected_output[16];
float channel_multipliers_in[16];
int channel_bias_terms_in[16];
int32_t mul_add_out[64];

int swap12(int i) {
    int bit12 = (i>>0) & 3;
    if (bit12 == 1 || bit12 == 2) {
        return i ^ 3;
    }
    return i;
}

int test_output_transform_fn_int16_kernel_transform(void) {
    otfn_int16_params_t otfn_params = {16};
    int errors = 0;
    for(int i = 0; i < 16; i++) {
        channel_multipliers_in[i] = (i+16)/32.0;
        channel_bias_terms_in[i] = 6*i-45;
        vDvR[swap12(i)   ] = i >= 8 ? -1 : 0;
        vDvR[swap12(i)+16] = i >= 8 ? (-103*i) : (101*i);
        expected_output[i] = round(vDvR[swap12(i)+16] * channel_multipliers_in[i]) + channel_bias_terms_in[i];
    }
    output_transform_fn_int16_kernel_transform(
        kernel_weights_in,
        channel_multipliers_in, channel_bias_terms_in,
        kernel_weights_out, mul_add_out,
        8, 16);

    output_transform_fn_int16(&otfn_params, vDvRoutput, vDvR, 0, mul_add_out);
    for(int i = 0; i < 16; i++) {
        if (vDvRoutput[i] != expected_output[i]) {
            errors++;
            printf("%d: %08x %08lx\n", i, vDvRoutput[i], expected_output[i]);
        }
    }
    
    return errors;
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += test_output_transform_fn_int16();
    errors += test_output_transform_fn_int16_kernel_transform();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
