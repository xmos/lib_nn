#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <tuple>
#include <vector>

#include "AggregateFn.hpp"
#include "vpu_sim.h"

using namespace nn;

/******************************
 * MaxPoolPatchFn
 *****************************/

MaxPoolPatchFn::Params::Params(const int32_t pixel_count)
    : pixel_count(pixel_count) {}

MaxPoolPatchFn::Params::Params(const nn::WindowGeometry &window)
    : pixel_count(window.shape.PixelCount()) {}

MaxPoolPatchFn::MaxPoolPatchFn(const Params *params) : params(params) {}

C_API
void maxpool_patch_ref(
    VPURingBuffer *A,  // This doesn't really make sense for maxpool.
    const int8_t *patch, const int pixels) {
  nn::VPU vpu;
  vpu_vector_t *curmax = (vpu_vector_t *)A->vR;
  vpu_vector_t vec_tmp;

  vpu.vsetc(MODE_S8);
  vpu.vldr(patch);
  vpu.vstr(curmax);

  int pix = pixels - 1;
  patch = &patch[VPU_INT8_EPV];

  while (pix--) {
    vpu.vldr(patch);
    vpu.vlsub(curmax);
    vpu.vdepth1();
    vpu.vstr(&vec_tmp);
    uint32_t mask = vec_tmp.u32[0];
    vpu.vldr(patch);
    vpu.vstrpv(curmax, mask);

    patch = &patch[VPU_INT8_EPV];
  }
}

void MaxPoolPatchFn::aggregate_fn(VPURingBuffer *acc, int8_t *input_patch,
                                  int32_t output_channel_group) {
#if defined(NN_USE_REF) || !defined(__XS3A__)
  maxpool_patch_ref(acc, input_patch, this->params->pixel_count);
#else
  maxpool_patch_xcore(acc, input_patch, this->params->pixel_count);
#endif
}

/******************************
 * MaxPoolDirectValidFn
 *****************************/

MaxPoolDirectValidFn::Params::Params(
    const maxpool_direct_valid_params &mp_params)
    : mp_params(mp_params) {}

MaxPoolDirectValidFn::Params::Params(const nn::ImageGeometry &input_img,
                                     const nn::WindowGeometry &window) {
  this->mp_params.col_stride = input_img.PixelBytes() * window.dilation.col;
  this->mp_params.cols = window.shape.width;
  this->mp_params.row_stride = input_img.GetStride(
      window.dilation.row, -(window.shape.width) * window.dilation.col, 0);
  this->mp_params.rows = window.shape.height;
}

MaxPoolDirectValidFn::MaxPoolDirectValidFn(const Params *params)
    : params(params) {}

C_API
void maxpool_direct_valid_ref(
    VPURingBuffer *A,  // This doesn't really make sense for maxpool.
    const int8_t *X, const maxpool_direct_valid_params *params) {
  nn::VPU vpu;
  vpu_vector_t *curmax = (vpu_vector_t *)A->vR;
  vpu_vector_t vec_tmp;

  vpu.vsetc(MODE_S8);
  vpu.vldd(X);
  vpu.vstd(curmax);

  for (auto row = params->rows; row; --row) {
    for (auto col = params->cols; col; --col) {
      vpu.vldr(X);
      vpu.vlsub(curmax);
      vpu.vdepth1();
      vpu.vstr(&vec_tmp);
      uint32_t mask = vec_tmp.u32[0];
      vpu.vldr(X);
      vpu.vstrpv(curmax, mask);

      X = &X[params->col_stride];
    }

    X = &X[params->row_stride];
  }
}

void MaxPoolDirectValidFn::aggregate_fn(VPURingBuffer *acc, int8_t *input_img,
                                        int32_t output_channel_group) {
#if defined(NN_USE_REF) || !defined(__XS3A__)
  maxpool_direct_valid_ref(acc, input_img, &this->params->mp_params);
#else
  maxpool_direct_valid_xcore(acc, input_img, &this->params->mp_params);
#endif
}
