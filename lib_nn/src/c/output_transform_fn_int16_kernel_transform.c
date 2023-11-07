#include <stdio.h>
#include <math.h>

#include "output_transform_fn_int16_kernel_transform.h"

// This defines the mapping of the output transform
static int ot_input_channel_used_for_output[16] = {
    0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15
};

static int aggr_input_channel_used_for_output[16] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
};

int output_transform_fn_int16_kernel_transform(
    int8_t *kernel_weights_in, float *channel_multipliers_in, int *channel_bias_terms_in,
    int8_t *kernel_weights_out, int32_t *mul_add_out,
    int input_channels, int output_channels) {
    for(int ochannel = 0; ochannel < output_channels; ochannel++) {
        int ochannel_group = ochannel & ~0xf;
        int ochannel_member = ochannel & 0xf;
        int model_ochannel = (ot_input_channel_used_for_output[ochannel_member] +
                              ochannel_group);
        int mul_add_major_index = ochannel_group * 2;
        int mul_index = 0;
        if (ochannel_member & 1) {
            mul_index += 0;
        } else {
            mul_index += 16;
        }
        mul_index += (ochannel_member >> 1) & 7;
        int add_index = mul_index + 8;
        mul_add_out[mul_index] = round(0x40000000 * channel_multipliers_in[ochannel]);
        mul_add_out[add_index] = channel_bias_terms_in[ochannel];
        for(int ichannel = 0; ichannel < input_channels; ichannel++) {
            int ichannel_group = ichannel & ~0xf;
            int ichannel_member = ichannel & 0xf;
            int in_index  = ochannel + ichannel * output_channels;
            int mapped_ochannel = ochannel_group + aggr_input_channel_used_for_output[ochannel_member];
            int out_index = mapped_ochannel + ichannel * output_channels;
            kernel_weights_out[out_index] = kernel_weights_in[in_index];
        }
    }
}

