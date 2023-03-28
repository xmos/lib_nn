#include <stdio.h>
#include <stdint.h>

#include "nn_op_utils.h"
#include "nn_operator.h"

void pad_3_to_4_prepare(uint32_t * n_3, 
    const unsigned height, 
    const unsigned width) {
    *n_3 = height*width;
}

/** Function that copies a single pixel of 3 bytes
 * @param outputs  pointer to the outputs array - incremented by 4 bytes
 * @param inputs   pointer to the input array - incremented by 3 bytes
 * @param N_3      number of pixels will be decremented by 1.
 */
static inline void pad_3_to_4_single(int8_t **outputs, int8_t **inputs, uint32_t *N_3, uint32_t pad_val) {
    for(uint32_t i = 0; i < 3; i++) {
        (*(int8_t**)outputs)[i] = (*inputs)[i];
    }
    (*(int8_t**)outputs)[3] = (int8_t)pad_val;
    *inputs += 3;
    *outputs += 4;
    *N_3 -= 1;
}

void pad_3_to_4_ref(int8_t outputs[], int8_t inputs[], uint32_t N_3, uint32_t pad_val){

    int8_t * output_p = (int8_t *)outputs;
    int8_t * input_p = (int8_t *)inputs;

    for(int i=0;i<N_3;i++){
        memcpy(output_p, input_p, 3);
        output_p += 3;
        input_p += 3;
        memcpy(output_p, &pad_val, 1);
        output_p += 1;
    }
}

#ifdef __XS3A__

/** Function that pads an image with 3-byte values with a 0.
 * This functions is highly optimised, but has constraints on the
 * alignment of the input image and on the number of bytes to be copied
 * Use ``pad_3_to_4`` to not have any constraints.
 *
 * The input image must be double word aligned.
 * The output image must be word aligned.
 * It copies the image in chunks of 24 bytes
 *
 * @param    outputs    output values, every word contains 3 bytes and a zero
 * @param    inputs     input values, RGBRGBRGBRGB...
 * @param    N_24       number of blocks of 24 bytes to copy
 *
 * @returns  The inner product
 */
extern void pad_3_to_4_asm(int32_t outputs[], int64_t inputs[], uint32_t N_24, uint32_t pad_val);

void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3, uint32_t pad_val) {
    // First copy single pixels until the input pointer is aligned
    // That will happen as it is incremented in steps of 3
    // But we may run out of pixels before it happens
    while((((uint32_t)inputs) & 7) != 0 && N_3 != 0) {
        pad_3_to_4_single(&outputs, &inputs, &N_3, pad_val);
    }
    
    // Now figure out whether the total number of pixels to be copied
    // Is a multiple of 24; if not, remember what the remainder is
    uint32_t tail_N_3 = N_3 & 7;    // remaining blocks of 3
    uint32_t N_24 = N_3 >> 3;       // Blocks of 24

    // Now copy the bulk of the data in blocks of 24
    if (N_24 != 0) {

#ifdef NN_USE_REF
        int8_t * outputs_p = (int8_t *)outputs;
        int8_t * inputs_p = inputs;
        for(uint32_t l=0;l<N_24;l++){
            for (unsigned i=0;i<8;i++){
                memcpy(outputs_p, inputs_p, 3);
                inputs_p += 3;
                outputs_p += 3;
                memcpy(outputs_p, &pad_val, 1);
                outputs_p += 1;
            }
        }
#else
        pad_3_to_4_asm((int32_t * )outputs, (int64_t *)inputs, N_24, pad_val);
#endif
    }

    // Finally, if there is a remainder, copy them a pixel at a time
    if (tail_N_3 != 0) {
        // Adjust the inputs and outputs pointer to point to the remainder.
        inputs +=  (N_24 << 3) * 3;
        outputs += (N_24 << 3) * 4;
        while(tail_N_3 != 0) {
            pad_3_to_4_single(&outputs, &inputs, &tail_N_3, pad_val);
        }
    }
}
#endif