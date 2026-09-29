// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

// This defines the mapping of the output transform multipliers from output channels
extern int ot_int16_mul_index_used_for_output[];

// This defines the mapping of the output transform biases from output channels
extern int ot_int16_add_index_used_for_output[];

// This defines the kernel mapping from output channels
extern int aggr_ot_int16_input_channel_used_for_output[];
