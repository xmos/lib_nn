// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_OPERATOR_H_
#define NN_OPERATOR_H_

#ifdef __XC__
extern "C" {
#endif

#include "nn_conv2d_bin.h"
#include "nn_conv2d_int8.h"
#include "nn_fully_connected.h"
#include "nn_layers.h"
#include "nn_op_utils.h"
#include "nn_pooling.h"

#ifdef __XC__
}  // extern "C"
#endif

#endif  // NN_OPERATOR_H_
