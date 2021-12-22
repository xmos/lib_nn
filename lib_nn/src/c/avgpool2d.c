// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_op_helper.h"
#include "nn_operator.h"
#include "xs3_vpu.h"

#ifndef AVGPOOL2D_INIT_ERROR_DETECTION_ENABLE
#define AVGPOOL2D_INIT_ERROR_DETECTION_ENABLE (1)
#endif

#if CONFIG_SYMMETRIC_SATURATION_avgpool2d
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

typedef struct {
  struct {
    struct {
      mem_stride_t row;
      mem_stride_t cog;
    } X;

    struct {
      mem_stride_t row;
      mem_stride_t col;
    } window;

    struct {
      mem_stride_t row;
      mem_stride_t cog;
    } Y;

  } stride;

  int32_t scale;
  uint32_t shift;

} nn_avgpool2d_job_t;

#undef NEG_SAT_VAL

#if CONFIG_SYMMETRIC_SATURATION_avgpool2d_global
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

void avgpool2d_global(nn_image_t* Y, const nn_image_t* X, const int32_t bias,
                      const int8_t scale, const uint16_t shift,
                      const nn_image_params_t* x_params) {
  avgpool2d_global_ext(Y, X, bias, scale, shift, x_params, 0,
                       x_params->channels, AVGPOOL2D_GLOBAL_FLAG_NONE);
}

void avgpool2d_global_ext_ref(nn_image_t* Y, const nn_image_t* X,
                              const int32_t bias, const int8_t scale,
                              const uint16_t shift,
                              const nn_image_params_t* x_params,
                              const unsigned chan_start,
                              const unsigned chan_count,
                              const nn_avgpool2d_global_flags_e flags) {
  Y = ADDR(Y, chan_start);
  X = ADDR(X, chan_start);

  const unsigned pix = x_params->height * x_params->width;

  for (unsigned ch = 0; ch < chan_count; ch++) {
    int32_t acc = bias;

    for (unsigned p = 0; p < pix; p++) {
      int32_t x = X[p * x_params->channels + ch];
      acc += x * scale;
    }

    Y[ch] = vlsat_single_s8(acc, shift, NEG_SAT_VAL, VPU_INT8_MAX);
  }
}

#undef NEG_SAT_VAL

#ifdef NN_USE_REF

void avgpool2d_gen(int8_t* Y, const int8_t* X,
                   const channel_count_t image_chans,
                   const nn_window_params_t* pooling_window,
                   const nn_window_op_job_params_t* job_params,
                   const nn_avgpool2d_flags_e flags,
                   const nn_avgpool2d_job_t* job) {
  avgpool2d_gen_ref(Y, X, image_chans, pooling_window, job_params, flags, job);
}

void avgpool2d_2x2(int8_t* Y, const int8_t* X,
                   const channel_count_t image_chans,
                   const nn_window_params_t* pooling_window,
                   const nn_window_op_job_params_t* job_params,
                   const nn_avgpool2d_flags_e flags,
                   const nn_avgpool2d_job_t* job) {
  avgpool2d_2x2_ref(Y, X, image_chans, pooling_window, job_params, flags, job);
}

void avgpool2d_global_ext(nn_image_t* Y, const nn_image_t* X,
                          const int32_t bias, const int8_t scale,
                          const uint16_t shift,
                          const nn_image_params_t* x_params,
                          const unsigned chan_start, const unsigned chan_count,
                          const nn_avgpool2d_global_flags_e flags) {
  avgpool2d_global_ext_ref(Y, X, bias, scale, shift, x_params, chan_start,
                           chan_count, flags);
}

#endif  // NN_USE_REF