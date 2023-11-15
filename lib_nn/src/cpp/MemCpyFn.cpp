#include "MemCpyFn.hpp"

#include "vpu_sim.h"

using namespace nn;

// TODO: [astew] CHAR_BIT not defined if I build with Cygwin or gcc+Ubuntu+WSL.
// Not in <limits> either.
#ifndef CHAR_BIT
#define CHAR_BIT (sizeof(char) * 8)
#endif

int DerefInputFn::get_scratch_bytes() { return 0; }
int DerefInputFn::get_overread_bytes() { return 0; }

DerefInputFn::DerefInputFn(const ImageGeometry &input,
                             const WindowGeometry &window)
    : p{input.GetStride(window.stride.row, 0, 0), input.GetStride(0, window.stride.col, 0)} {}


int8_t *nn::memcpyfn_deref(const memcpyfn_deref_params_t *params, int8_t *T, int8_t *X, int32_t output_v_coord,
                                 int32_t output_h_coord,
                                 int32_t output_c_coord) {
  return X + (int)(output_v_coord * params->bytes_per_h_line +
                   output_h_coord * params->bytes_per_pixel + output_c_coord);
}

int ImToColPadded::get_scratch_bytes() {
  return p.kernel_height * p.kernel_width *
             p.bytes_per_copy_per_channel +
         XS3_VPU_VREG_WIDTH_BYTES;
}

int ImToColPadded::get_overread_bytes() {
  return XS3_VPU_VREG_WIDTH_BYTES;  // TODO this will be defined by the
                                    // implementation of memcpy
}

ImToColPadded::ImToColPadded(const ImageGeometry &X, const WindowGeometry &K,
                              const padding_t &padding,
                              const int input_ch_per_output,
                              const int8_t pad_val) {
  p.kernel_height = K.shape.height;
  p.kernel_width = K.shape.width;

  p.vertical_stride = K.stride.row;
  p.horizontal_stride = K.stride.col;
  p.vertical_dilation = K.dilation.row;
  p.horizontal_dilation = K.dilation.col;

  p.input_v_length = X.height;
  p.input_h_length = X.width;

  p.padding_val = pad_val;

  p.bytes_per_pixel = X.PixelBytes();
  p.bytes_per_h_line = X.RowBytes();

  p.padding_top = padding.top;
  p.padding_left = padding.left;

  p.bytes_per_copy_per_channel = (input_ch_per_output * CHAR_BIT) / CHAR_BIT;

  p.x_h_mem_stride = p.bytes_per_pixel * p.horizontal_dilation;
  p.x_v_mem_stride = p.vertical_dilation * p.bytes_per_h_line -
                   p.bytes_per_pixel * p.horizontal_dilation * p.kernel_width;
}

ImToColPadded::ImToColPadded(const Filter2dGeometry &filter,
                              const int8_t pad_val,
                              const int channels_per_output_group) {
  // This constructor is only intended to be used with a depthwise filter.
  // See the note below.
  assert(filter.window.shape.depth == 1);

  p.kernel_height = filter.window.shape.height;
  p.kernel_width = filter.window.shape.width;

  p.vertical_stride = filter.window.stride.row;
  p.horizontal_stride = filter.window.stride.col;
  p.vertical_dilation = filter.window.dilation.row;
  p.horizontal_dilation = filter.window.dilation.col;

  p.padding_top = filter.Padding().top;
  p.padding_left = filter.Padding().left;

  p.input_v_length = filter.input.height;
  p.input_h_length = filter.input.width;

  p.padding_val = pad_val;

  p.bytes_per_pixel = filter.input.PixelBytes();
  p.bytes_per_h_line = filter.input.RowBytes();

  // horizontal_mem_stride = filter.input.RowBytes();

  p.x_h_mem_stride = p.bytes_per_pixel * p.horizontal_dilation;
  p.x_v_mem_stride = p.vertical_dilation * p.bytes_per_h_line -
                   p.bytes_per_pixel * p.horizontal_dilation * p.kernel_width;

  /// NOTE: In a dense (non-depthwise) filter, the entirety of each input pixel
  /// goes into the
  ///       patch buffer, because the same input channels (all of them) are
  ///       needed to compute every output channel. So in that case,
  ///       bytes_per_copy_per_channel is basically just the size of an input
  ///       pixel. But the key point is that the range of channels we're copying
  ///       into the patch DOES NOT DEPEND on how many output channels we
  ///       compute in parallel (if any at all). In a depthwise filter each
  ///       output channel relies only on exactly 1 input channel. So we're
  ///       doing something that looks similar but is conceptually very
  ///       different. In the depthwise case, we're putting more than 1 channel
  ///       in the patch buffer because the range of chanels we're copying into
  ///       the patch DOES DEPEND on how many output channels we compute in
  ///       parallel. The 'depth' parameter of the window shape describes how
  ///       many input channels are needed to compute an output. It is UNRELATED
  ///       to the number of channels we compute in parallel. So we need to
  ///       multiply by the degree of parallelism of the operator.
  p.bytes_per_copy_per_channel =
      (channels_per_output_group * filter.window.shape.depth *
       filter.window.shape.element_bits) /
      CHAR_BIT;
}

int8_t *memcpyfn_imtocol_padded_impl(const memcpyfn_imtocol_padded_params_t *params, int8_t *T, int8_t *X,
                                       int32_t output_v_coord,
                                       int32_t output_h_coord,
                                       int32_t output_c_coord) {
  int8_t *T_in = T;

  // extract all of these
  int32_t input_v_coord =
      output_v_coord * params->vertical_stride - params->padding_top;
  int32_t strided_h_coord =
      output_h_coord * params->horizontal_stride - params->padding_left;
  int8_t *X_cur_p =
      X + (int)(strided_h_coord * params->bytes_per_pixel + output_c_coord +
                input_v_coord * params->bytes_per_h_line);

  for (int32_t k_height = 0; k_height < params->kernel_height; k_height++) {
    int p = input_v_coord < 0;
    p |= input_v_coord >= params->input_v_length;

    int32_t input_h_coord = strided_h_coord;

    for (int32_t k_width = 0; k_width < params->kernel_width; k_width++) {
      int q = p;
      q |= input_h_coord < 0;
      q |= input_h_coord >= params->input_h_length;

      if (q) {
        memset(T, params->padding_val, params->bytes_per_copy_per_channel);
      } else {
        memcpy(T, X_cur_p, params->bytes_per_copy_per_channel);
      }

      T += params->bytes_per_copy_per_channel;

      X_cur_p += params->x_h_mem_stride;

      input_h_coord += params->horizontal_dilation;
    }
    input_v_coord += params->vertical_dilation;

    X_cur_p += params->x_v_mem_stride;
  }

  // Write padding to the tail, zeros is fastest
  memset(T, 0, XS3_VPU_VREG_WIDTH_BYTES);

  return T_in;
}

/*
This constructor is used for testing
*/
ImToColValid::ImToColValid(const ImageGeometry &X, const WindowGeometry &K,
                             const int input_ch_per_output, const bool dontzero) {
  int bytes_per_copy_per_channel =
      (input_ch_per_output * X.element_bits) / CHAR_BIT;

  p.bytes_per_pixel = X.PixelBytes();
  p.bytes_per_h_line = X.RowBytes();

  assert(X.RowBytes() == X.width * p.bytes_per_pixel);

  // This is the amount to copy in vpu words (round up)
  p.input_channel_groups =
      (bytes_per_copy_per_channel + XS3_VPU_VREG_WIDTH_BYTES - 1) /
      XS3_VPU_VREG_WIDTH_BYTES;

  p.input_channel_groups -= 1;

  int bytes_actually_copied =
      (p.input_channel_groups + 1) * XS3_VPU_VREG_WIDTH_BYTES;
  p.T_rewind = bytes_actually_copied - bytes_per_copy_per_channel - 32;
  uint32_t bitsleft = bytes_per_copy_per_channel & (XS3_VPU_VREG_WIDTH_BYTES - 1);
  if (bitsleft != 0) {
    p.T_vstrpv_mask = (1ULL << bitsleft) -1;
  } else {
    p.T_vstrpv_mask = (1ULL << XS3_VPU_VREG_WIDTH_BYTES) -1;
  }
  p.T_dontzero = dontzero;
  p.input_height = K.shape.height;
  p.input_height -= 1;
  p.input_width = K.shape.width;
  p.input_width -= 1;

  p.horizontal_mem_stride =
      p.bytes_per_pixel * K.dilation.col - bytes_actually_copied;
  p.vertical_mem_stride = p.bytes_per_h_line * K.dilation.row -
                        (p.input_width + 1) * p.bytes_per_pixel * K.dilation.col;

  // TODO rename these to account for the multiplication of strides
  p.bytes_per_h_line *= K.stride.row;
  p.bytes_per_pixel *= K.stride.col;
}

int ImToColValid::get_scratch_bytes() {
  return (p.input_height + 1) * (p.input_width + 1) *
             ((p.input_channel_groups + 1) * XS3_VPU_VREG_WIDTH_BYTES -
              (p.T_rewind + 32)) +
         XS3_VPU_VREG_WIDTH_BYTES;
}

int ImToColValid::get_overread_bytes() { return p.T_rewind + 32; }

int8_t *memcpyfn_imtocol_valid_impl(const memcpyfn_imtocol_valid_params_t *params, int8_t *T, int8_t *X,
                                      int32_t output_v_coord,
                                      int32_t output_h_coord,
                                      int32_t output_c_coord) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  int8_t *X_cur_p =
      X + (int)(output_v_coord * params->bytes_per_h_line +
                output_h_coord * params->bytes_per_pixel + output_c_coord);

  int8_t *T_in = T;
  uint32_t mask = params->T_vstrpv_mask;
  for (int32_t i_height = params->input_height; i_height >= 0; i_height--) {
    for (int32_t i_width = params->input_width; i_width >= 0; i_width--) {
      // This loop copies a whole pixel
      for (int32_t i_ch_group = params->input_channel_groups; i_ch_group > 0;
           i_ch_group--) {
        VLDD(vpu, X_cur_p);
        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        VSTD(vpu, T);
        T += XS3_VPU_VREG_WIDTH_BYTES;
      }
      VLDR(vpu, X_cur_p);
      X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

      VSTRPV(vpu, T, mask);

      T -= params->T_rewind;

      // Advance the X_cur_p to the start of the next horizontal pixel
      X_cur_p += params->horizontal_mem_stride;
    }

    // Advance the X_cur_p to the start of the next vertical pixel
    X_cur_p += params->vertical_mem_stride;
  }

  if (!params->T_dontzero) {
    VCLRDR(vpu);  // Write padding to the tail, zeros is fastest
    VSTD(vpu, T);
  }

  return T_in;
}

extern "C" int8_t *im_to_col_valid_impl_asm(const memcpyfn_imtocol_valid_params_t *params, int8_t *T, int8_t *X,
                                            int32_t output_v_coord,
                                            int32_t output_h_coord,
                                            int32_t output_c_coord);
int8_t *nn::memcpyfn_imtocol_valid(const memcpyfn_imtocol_valid_params_t *params, int8_t *T, int8_t *X, int32_t output_v_coord,
                                 int32_t output_h_coord,
                                 int32_t output_c_coord) {
#ifdef NN_USE_REF
  return memcpyfn_imtocol_valid_impl(params, T, X, output_v_coord, output_h_coord, output_c_coord);
#else
  return im_to_col_valid_impl_asm(params, T, X, output_v_coord,
                                  output_h_coord, output_c_coord);
#endif  // NN_USE_REF
}

int8_t *nn::memcpyfn_imtocol_padded(const memcpyfn_imtocol_padded_params_t *params, int8_t *T, int8_t *X, int32_t output_v_coord,
                                  int32_t output_h_coord,
                                  int32_t output_c_coord) {
  return memcpyfn_imtocol_padded_impl(params, T, X, output_v_coord, output_h_coord, output_c_coord);
}
