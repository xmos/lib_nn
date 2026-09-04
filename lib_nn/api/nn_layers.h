// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_LAYERS_H_
#define NN_LAYERS_H_

#include "add_elementwise.h"
#include "add_int16.h"
#include "add_int16_transform.h"
#include "argmax.h"
#include "bsign.h"
#include "dequantize_int16.h"
#include "dequantize_int16_transform.h"
#include "expand_8_to_16.h"
#include "lookup8.h"
#include "mean.h"
#include "mul.h"
#include "multiply_int16.h"
#include "multiply_int16_transform.h"
#include "output_transform_fn_int16.h"
#include "output_transform_fn_int16_kernel_transform.h"
#include "output_transform_fn_int16_mappings.h"
#include "pad.h"
#include "quantize_int16.h"
#include "quantize_int16_transform.h"
#include "quadratic_approximation.h"
#include "quadratic_interpolation.h"
#include "requantize.h"
#include "softmax.h"

#endif  // NN_LAYERS_H_
