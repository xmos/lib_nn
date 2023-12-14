#include "output_transform_fn_int16_mappings.h"

// This defines the mapping of the output transform multipliers from output channels
// Ie, in order to calculate OT[3] you need to use mul_add[17]
int ot_int16_mul_index_used_for_output[16] = {
    16, 0, 17, 1, 18, 2, 19, 3, 20, 4, 21, 5, 22, 6, 23, 7
};

// This defines the mapping of the output transform biases from output channels
// Ie, in order to calculate OT[3] you need to use mul_add[25]
int ot_int16_add_index_used_for_output[16] = {
    24, 8, 25, 9, 26, 10, 27, 11, 28, 12, 29, 13, 30, 14, 31, 15
};

// This defines the kernel mapping from output channels
int aggr_ot_int16_input_channel_used_for_output[16] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
};
