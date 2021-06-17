#ifndef LIB_NN_MEMCPY_FN_HPP_
#define LIB_NN_MEMCPY_FN_HPP_

#include "geom/Filter2dGeometry.hpp"

namespace nn {

/**
 * Interface implemented by all patch handlers.
 */
class MemCpyFn {
 public:
  /**
   * Copy the relevant region of the input image to the patch buffer, and return
   * a pointer to where the aggregation handler should begin consuming inputs.
   *
   * @TODO: astew: The `c` below is confusing, since `h` and `w` are in the
   * output image's coordinate space, one is tempted to thing `c` also is, but
   * `c` is actually in the input image's coordinate space. e.g. for
   * non-depthwise mem copies `c` will always be zero (because we copy all input
   * channels), and for depthwise mem copies while it's the output channel
   * coordinate that is given, that only works because in a depthwise op the
   * output and input channel indices are the same. I suggest that memcpy_fn()
   * should always be given the current output channel (first of the current
   * output channel group), and internally any MemCpyFn implementation that is
   * supposed to work for both depthwise and non-depthwise ops should just keep
   * a 'channel stride' parameter (@see nn::WindowGeometry) that will be either
   * 0 (for dense convolutions) or 1 (for depthwise convolutions). This approach
   * will also generalize if we ever end up with convolutions in which, for
   * example, output channel 0 is a function of both input channels 0 and 1 (but
   * no others), and output channel 1 is a function of both input channels 2 and
   * 3 (but no others), etc, by using a channel stride value of 2.
   *
   * @param [inout] T   pointer to patch buffer
   * @param [in] X      pointer to base address of the input image
   * @param [in] h      row index of the _OUTPUT_ pixel being processed
   * @param [in] w      column index of the _OUTPUT_ pixel being processed
   * @param [in] c      channel index of the _INPUT_ image from which to start
   * copying elements
   */
  virtual int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w,
                             int32_t c = 0) = 0;

  /**
   * @brief Get the number of bytes required for the scratch memory to hold the
   * result of the mem copy. This accounts for the VPU memory read and write
   * limitations.
   *
   * @return int
   */
  virtual int get_scratch_bytes() = 0;

  /**
   * @brief Get the number of bytes that VPU will overread from source tensor.
   * This is to account for vector reads being used where a sub-vector read is
   * required. This is used to prevent OOB memory accesses, i.e. if a memory
   * location within this much beyond the end of an input tensor is invalid then
   * it should be handled gracefully.
   *
   * @return int
   */
  virtual int get_overread_bytes() = 0;
};

/**
 *
 */
class DerefInputFn : public MemCpyFn {
 public:
  class Params {
   public:
    int32_t bytes_per_h_line;
    int32_t bytes_per_pixel;

    /**
     * @brief Construct a new Params object
     *
     * @param bytes_per_h_line Count of bytes in a horizontal line of the input
     * tensor.
     * @param bytes_per_pixel Count of bytes in a pixel of the input tensor.
     */
    Params(int32_t bytes_per_h_line, int32_t bytes_per_pixel);

    /**
     * @brief Construct a new Params object
     *
     * @param X Class describing the properties of the input tensor over which
     * the convolution will be performed over.
     * @param K Class describing the properties of the convolution to be
     * performed.
     */
    Params(const ImageGeometry &X, const WindowGeometry &K);

    /**
     * @brief Construct a new Params object
     *
     * @param filter_geometry Class representing the properties of the input
     * tensor and convolution properties over which the convolution will be
     * performed.
     */
    Params(const Filter2dGeometry &filter_geometry);

    Params(std::istream &stream);

    void Serialize(std::ostream &stream) const;
  };
  /**
   * @brief This describes the region over which this class will perform its
   * operation(Memcopy).
   */
  Params *params;

 public:
  DerefInputFn(Params *params) : params(params){};
  int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c);
  int get_scratch_bytes();
  int get_overread_bytes();
};

/**
 *
 */
class ImToColPadded : public MemCpyFn {
 public:
  class Params {
   public:
    int32_t kernel_height;
    int32_t input_v_length;

    int32_t kernel_width;
    int32_t input_h_length;

    int32_t x_h_mem_stride;
    int32_t horizontal_dilation;

    int32_t x_v_mem_stride;
    int32_t vertical_dilation;

    int32_t bytes_per_copy_per_channel;
    int32_t padding_val;

    // these are outside the loop
    int32_t bytes_per_h_line;
    int32_t bytes_per_pixel;
    int32_t padding_left;
    int32_t padding_top;
    int32_t vertical_stride;
    int32_t horizontal_stride;

   public:
    /**
     * @brief Construct a new Params object
     *
     * @param X Class describing the properties of the input tensor over which
     * the convolution will be performed over.
     * @param K Class describing the properties of the convolution to be
     * performed.
     * @param padding Struct describing the padding to be applied during the
     * copy.
     * @param input_ch_per_output The count of input channeks that contribute to
     * an output channel. For example, a depthwise convolution will have one
     * input channel per output channel whereas a Conv2D will most likely have
     * the input tensors channel count.
     * @param padding_value The value to insert for the padding.
     */
    Params(const ImageGeometry &X, const WindowGeometry &K,
           const padding_t &padding, const int input_ch_per_output,
           const int8_t padding_value);

    /**
     * @brief Construct a new Params object
     *
     * @param filter_geometry Class representing the properties of the input
     * tensor and convolution properties over which the convolution will be
     * performed.
     * @param padding_value The value to insert for the padding.
     * @param input_ch_per_output The count of input channeks that contribute to
     * an output channel. For example, a depthwise convolution will have one
     * input channel per output channel whereas a Conv2D will most likely have
     * the input tensors channel count.
     */
    Params(const Filter2dGeometry &filter_geometry, const int8_t padding_value,
           const int input_ch_per_output);

    Params(std::istream &stream);

    void Serialize(std::ostream &stream) const;
  };

 private:
  /**
   * @brief This describes the region over which this class will perform its
   * operation(Memcopy).
   */
  Params *params;

 public:
  ImToColPadded(Params *p) : params(p) {}
  int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c);
  int get_scratch_bytes();
  int get_overread_bytes();

 private:
  int8_t *memcopy_fn_impl(int8_t *T, int8_t *X, int32_t h, int32_t w,
                          int32_t c);
};

class ImToColValid : public MemCpyFn {
 public:
  struct Params {
    /**
     * Bytes per row of the input image
     */
    int32_t bytes_per_h_line;

    /**
     * Bytes per pixels of the filter window.
     */
    int32_t bytes_per_pixel;

    /**
     * Height of the filter window in pixels.
     */
    int32_t input_height;

    /**
     * Width of the filter window in pixels.
     */
    int32_t input_width;

    /**
     * The number of VPU words (vectors) to copy for the entire filter window.
     *
     * Note that this should be rounded up if the number of words is not
     * integral.
     */
    int32_t input_channel_groups;

    /**
     * The difference between the number of bytes actually copied and the target
     * number of bytes to copy.
     */
    int32_t T_rewind;

    // The bytes to inc an X pointer by to move by one horizontal stride
    int32_t horizontal_mem_stride;

    // The bytes to inc an X pointer by to move by one vertical stride
    // and horizontally bacwards by the kernel width.
    // i.e. from X[h][w + kernel_width - 1] to X[h+1][w].
    int32_t vertical_mem_stride;

    /**
     * @brief Construct a new Params object
     *
     * @param X Class describing the properties of the input tensor over which
     * the convolution will be performed over.
     * @param K Class describing the properties of the convolution to be
     * performed.
     * @param input_ch_per_output The count of input channeks that contribute to
     * an output channel. For example, a depthwise convolution will have one
     * input channel per output channel whereas a Conv2D will most likely have
     * the input tensors channel count.
     */
    Params(const ImageGeometry &X, const WindowGeometry &K,
           const int input_ch_per_output);
  };

 private:
  /**
   * @brief This describes the region over which this class will perform its
   * operation(Memcopy).
   */
  Params *params;

 public:
  // input_ch_per_output lets the kernel know how many input channels to copy to
  // scratch
  ImToColValid(Params *params) : params(params){};

  int get_scratch_bytes();
  int get_overread_bytes();

  int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c);

 private:
  int8_t *memcopy_fn_impl(int8_t *T, int8_t *X, int32_t h, int32_t w,
                          int32_t c);
};
}  // namespace nn
#endif  // LIB_NN_MEMCPY_FN_HPP_
