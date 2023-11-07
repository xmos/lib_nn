#include "output_transform_fn_int16_mappings.h"

// This defines the mapping of the output transform vDvR from output channels
// Ie, in order to calculate OT[2] you need to use vDvR[1]
int ot_int16_input_channel_used_for_output[16] = {
    0, 2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 14, 13, 15
};

// This defines the mapping of the output transform multipliers from output channels
// Ie, in order to calculate OT[3] you need to use mul_add[17]
int ot_int16_mul_index_used_for_output[16] = {
    0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23
};

// This defines the mapping of the output transform biases from output channels
// Ie, in order to calculate OT[3] you need to use mul_add[25]
int ot_int16_add_index_used_for_output[16] = {
    8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31
};

// This defines the kernel mapping from output channels
int aggr_ot_int16_input_channel_used_for_output[16] = {
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0
};
