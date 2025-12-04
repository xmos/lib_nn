#include <stdio.h>
#include <stdint.h>


/** Function that pads an image with 1-byte values with a padding value
 * This functions is highly optimised, and expects the two pointers to be word aligned.
 * It copies the image in chunks of 4 bytes
 *
 * @param    outputs    output values, every word contains 1 byte, three bytes padding
 * @param    inputs     input values, GGGGGGGG
 * @param    N          number of 4-byte chunks to copy
 *
 * @returns  The inner product
 */
extern void pad_1_to_4_asm(int32_t outputs[], int32_t inputs[], uint32_t N_24, uint32_t pad_val);

void pad_1_to_4_ref(int8_t outputs[], int8_t inputs[], uint32_t N, uint32_t pad_val){

    uint32_t * output_p = (uint32_t *)outputs;
    uint8_t * input_p = (uint8_t *)inputs;

    for(int i=0;i<N*4;i++){
        *output_p = *input_p | (pad_val & 0xffffff00);
        output_p += 1;
        input_p += 1;
    }
}

void pad_1_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N, uint32_t pad_val) {
#if defined(NN_USE_REF) || defined(__riscv_xxcore)
        pad_1_to_4_ref(outputs, inputs, N, pad_val);
#else
        pad_1_to_4_asm((int32_t *) outputs, (int32_t *)inputs, N, pad_val);
#endif
}

#ifdef PAD_1_TO_4_MAIN

int main(void) {
    int input[2];
    int output0[8];
    int output1[8];
    int output_ref[8] = {0xAAAAAF01,0xAAAAAF02,0xAAAAAF03,0xAAAAAF04,0xAAAAAF05,0xAAAAAF06,0xAAAAAF07,0xAAAAAF08};
    ((uint8_t *)input)[0] = 1;
    ((uint8_t *)input)[1] = 2;
    ((uint8_t *)input)[2] = 3;
    ((uint8_t *)input)[3] = 4;
    ((uint8_t *)input)[4] = 5;
    ((uint8_t *)input)[5] = 6;
    ((uint8_t *)input)[6] = 7;
    ((uint8_t *)input)[7] = 8;
    pad_1_to_4_ref((int8_t *) output0, (int8_t *)input, 2, 0xAAAAAFAA);
    pad_1_to_4_asm((int32_t *)output1, (int32_t *)input, 2, 0xAAAAAFAA);
    for(int i = 0; i < 8; i++) {
        if (output0[i] != output1[i] || output0[i] != output_ref[i]) {
            printf("Error %08x %08x %08x\n", output0[i], output1[i], output_ref[i]);
        }
    }
}

#endif
