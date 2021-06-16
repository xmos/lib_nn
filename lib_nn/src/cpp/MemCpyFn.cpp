#include "MemCpyFn.hpp"

#include "vpu_sim.h"

using namespace nn;

// TODO: [astew] CHAR_BIT not defined if I build with Cygwin or gcc+Ubuntu+WSL.
// Not in <limits> either.
#ifndef CHAR_BIT
#define CHAR_BIT (sizeof(char) * 8)
#endif

DerefInputFn::Params::Params(const int32_t bytes_per_h_line,
                             const int32_t bytes_per_pixel)
    : bytes_per_h_line(bytes_per_h_line), bytes_per_pixel(bytes_per_pixel) {}

DerefInputFn::Params::Params(const ImageGeometry &input,
                             const WindowGeometry &window)
    : bytes_per_h_line(input.GetStride(window.stride.row, 0, 0)),
      bytes_per_pixel(input.GetStride(0, window.stride.col, 0)) {}

DerefInputFn::Params::Params(const Filter2dGeometry &filter)
    : Params(filter.input, filter.window) {}

DerefInputFn::Params::Params(std::istream &stream) {
#define READ_MEMBER(MEMBER) \
  stream.read(reinterpret_cast<char *>(&this->MEMBER), sizeof(this->MEMBER))

  READ_MEMBER(bytes_per_h_line);
  READ_MEMBER(bytes_per_pixel);

#undef READ_MEMBER
}

void DerefInputFn::Params::Serialize(std::ostream &stream) const {
#define WRITE_MEMBER(MEMBER)                                  \
  stream.write(reinterpret_cast<const char *>(&this->MEMBER), \
               sizeof(this->MEMBER))

  WRITE_MEMBER(bytes_per_h_line);
  WRITE_MEMBER(bytes_per_pixel);

#undef WRITE_MEMBER
}

int DerefInputFn::get_scratch_bytes() { return 0; }
int DerefInputFn::get_overread_bytes() { return 0; }

int8_t *DerefInputFn::memcopy_fn(int8_t *T, int8_t *X, int32_t output_v_coord,
                                 int32_t output_h_coord,
                                 int32_t output_c_coord) {
  return X + (int)(output_v_coord * params->bytes_per_h_line +
                   output_h_coord * params->bytes_per_pixel + output_c_coord);
}

int ImToColPadded::get_scratch_bytes() {
  return params->kernel_height * params->kernel_width *
             params->bytes_per_copy_per_channel +
         XS3_VPU_VREG_WIDTH_BYTES;
}

int ImToColPadded::get_overread_bytes() {
  return XS3_VPU_VREG_WIDTH_BYTES;  // TODO this will be defined by the
                                    // implementation of memcpy
}

ImToColPadded::Params::Params(const ImageGeometry &X, const WindowGeometry &K,
                              const padding_t &padding,
                              const int input_ch_per_output,
                              const int8_t pad_val) {
  kernel_height = K.shape.height;
  kernel_width = K.shape.width;

  vertical_stride = K.stride.row;
  horizontal_stride = K.stride.col;
  vertical_dilation = K.dilation.row;
  horizontal_dilation = K.dilation.col;

  input_v_length = X.height;
  input_h_length = X.width;

  padding_val = pad_val;

  bytes_per_pixel = X.PixelBytes();
  bytes_per_h_line = X.RowBytes();

  padding_top = padding.top;
  padding_left = padding.left;

  bytes_per_copy_per_channel = (input_ch_per_output * CHAR_BIT) / CHAR_BIT;

  x_h_mem_stride = bytes_per_pixel * horizontal_dilation;
  x_v_mem_stride = vertical_dilation * bytes_per_h_line -
                   bytes_per_pixel * horizontal_dilation * kernel_width;
}

ImToColPadded::Params::Params(const Filter2dGeometry &filter,
                              const int8_t pad_val,
                              const int channels_per_output_group) {
  // This constructor is only intended to be used with a depthwise filter.
  // See the note below.
  assert(filter.window.shape.depth == 1);

  kernel_height = filter.window.shape.height;
  kernel_width = filter.window.shape.width;

  vertical_stride = filter.window.stride.row;
  horizontal_stride = filter.window.stride.col;
  vertical_dilation = filter.window.dilation.row;
  horizontal_dilation = filter.window.dilation.col;

  padding_top = filter.Padding().top;
  padding_left = filter.Padding().left;

  input_v_length = filter.input.height;
  input_h_length = filter.input.width;

  padding_val = pad_val;

  bytes_per_pixel = filter.input.PixelBytes();
  bytes_per_h_line = filter.input.RowBytes();

  // horizontal_mem_stride = filter.input.RowBytes();

  x_h_mem_stride = bytes_per_pixel * horizontal_dilation;
  x_v_mem_stride = vertical_dilation * bytes_per_h_line -
                   bytes_per_pixel * horizontal_dilation * kernel_width;

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
  bytes_per_copy_per_channel = channels_per_output_group *
                               filter.window.shape.depth *
                               filter.window.shape.channel_depth;
}

ImToColPadded::Params::Params(std::istream &stream) {
#define READ_MEMBER(MEMBER) \
  stream.read(reinterpret_cast<char *>(&this->MEMBER), sizeof(this->MEMBER))

  READ_MEMBER(kernel_height);
  READ_MEMBER(kernel_width);
  READ_MEMBER(vertical_stride);
  READ_MEMBER(horizontal_stride);
  READ_MEMBER(vertical_dilation);
  READ_MEMBER(horizontal_dilation);
  READ_MEMBER(padding_left);
  READ_MEMBER(padding_top);
  READ_MEMBER(input_v_length);
  READ_MEMBER(input_h_length);
  READ_MEMBER(padding_val);
  READ_MEMBER(bytes_per_h_line);
  READ_MEMBER(bytes_per_pixel);
  READ_MEMBER(bytes_per_copy_per_channel);

#undef READ_MEMBER
}

void ImToColPadded::Params::Serialize(std::ostream &stream) const {
#define WRITE_MEMBER(MEMBER)                                  \
  stream.write(reinterpret_cast<const char *>(&this->MEMBER), \
               sizeof(this->MEMBER))

  WRITE_MEMBER(kernel_height);
  WRITE_MEMBER(kernel_width);
  WRITE_MEMBER(vertical_stride);
  WRITE_MEMBER(horizontal_stride);
  WRITE_MEMBER(vertical_dilation);
  WRITE_MEMBER(horizontal_dilation);
  WRITE_MEMBER(padding_left);
  WRITE_MEMBER(padding_top);
  WRITE_MEMBER(input_v_length);
  WRITE_MEMBER(input_h_length);
  WRITE_MEMBER(padding_val);
  WRITE_MEMBER(bytes_per_h_line);
  WRITE_MEMBER(bytes_per_pixel);
  WRITE_MEMBER(bytes_per_copy_per_channel);

#undef WRITE_MEMBER
}

int8_t *ImToColPadded::memcopy_fn_impl(int8_t *T, int8_t *X,
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
  memset(T, 0, 32);

  return T_in;
}

/*
This constructor is used for testing
*/
ImToColValid::Params::Params(const ImageGeometry &X, const WindowGeometry &K,
                             const int input_ch_per_output) {
  // TODO
  // int bytes_per_copy_per_channel = (input_ch_per_output * X.bits_per_element)
  // / CHAR_BIT;
  int bytes_per_copy_per_channel = (input_ch_per_output * CHAR_BIT) / CHAR_BIT;

  bytes_per_pixel = X.PixelBytes();
  bytes_per_h_line = X.RowBytes();

  assert(X.RowBytes() == X.width * bytes_per_pixel);

  // This is the amount to copy in vpu words (round up)
  input_channel_groups =
      (bytes_per_copy_per_channel + XS3_VPU_VREG_WIDTH_BYTES - 1) /
      XS3_VPU_VREG_WIDTH_BYTES;
  input_channel_groups -= 1;

  int bytes_actually_copied =
      (input_channel_groups + 1) * XS3_VPU_VREG_WIDTH_BYTES;
  T_rewind = bytes_actually_copied - bytes_per_copy_per_channel;

  input_height = K.shape.height;
  input_height -= 1;
  input_width = K.shape.width;
  input_width -= 1;

  horizontal_mem_stride =
      bytes_per_pixel * K.dilation.col - bytes_actually_copied;
  vertical_mem_stride = bytes_per_h_line * K.dilation.row -
                        (input_width + 1) * bytes_per_pixel * K.dilation.col;

  // TODO rename these to account for the multiplication of strides
  bytes_per_h_line *= K.stride.row;
  bytes_per_pixel *= K.stride.col;
}

int ImToColValid::get_scratch_bytes() {
  return (params->input_height + 1) * (params->input_width + 1) *
             ((params->input_channel_groups + 1) * XS3_VPU_VREG_WIDTH_BYTES -
              params->T_rewind) +
         XS3_VPU_VREG_WIDTH_BYTES;
}

int ImToColValid::get_overread_bytes() { return params->T_rewind; }

int8_t *ImToColValid::memcopy_fn_impl(int8_t *T, int8_t *X,
                                      int32_t output_v_coord,
                                      int32_t output_h_coord,
                                      int32_t output_c_coord) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  int8_t *X_cur_p =
      X + (int)(output_v_coord * params->bytes_per_h_line +
                output_h_coord * params->bytes_per_pixel + output_c_coord);

  int8_t *T_in = T;

  for (int32_t i_height = params->input_height; i_height >= 0; i_height--) {
    for (int32_t i_width = params->input_width; i_width >= 0; i_width--) {
      // This loop copies a whole pixel
      for (int32_t i_ch_group = params->input_channel_groups; i_ch_group >= 0;
           i_ch_group--) {
        VLDD(vpu, X_cur_p);
        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        VSTD(vpu, T);
        T += XS3_VPU_VREG_WIDTH_BYTES;
      }

      T -= params->T_rewind;

      // Advance the X_cur_p to the start of the next horizontal pixel
      X_cur_p += params->horizontal_mem_stride;
    }

    // Advance the X_cur_p to the start of the next vertical pixel
    X_cur_p += params->vertical_mem_stride;
  }

  // Write padding to the tail, zeros is fastest
  VCLRDR(vpu);
  VSTD(vpu, T);

  return T_in;
}

extern "C" int8_t *im_to_col_valid_impl_asm(void *params, int8_t *T, int8_t *X,
                                            int32_t output_v_coord,
                                            int32_t output_h_coord,
                                            int32_t output_c_coord);
int8_t *ImToColValid::memcopy_fn(int8_t *T, int8_t *X, int32_t output_v_coord,
                                 int32_t output_h_coord,
                                 int32_t output_c_coord) {
#ifdef NN_USE_REF
  return memcopy_fn_impl(T, X, output_v_coord, output_h_coord, output_c_coord);
#else
  return im_to_col_valid_impl_asm(this->params, T, X, output_v_coord,
                                  output_h_coord, output_c_coord);
#endif  // NN_USE_REF
}

int8_t *ImToColPadded::memcopy_fn(int8_t *T, int8_t *X, int32_t output_v_coord,
                                  int32_t output_h_coord,
                                  int32_t output_c_coord) {
  return memcopy_fn_impl(T, X, output_v_coord, output_h_coord, output_c_coord);
}
