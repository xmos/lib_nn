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
  int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w,
                             int32_t c = 0);

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

struct memcpyfn_deref_params_t {
    int32_t bytes_per_h_line;
    int32_t bytes_per_pixel;
};

/**
 *
 */
class DerefInputFn : public MemCpyFn {
  private:
  memcpyfn_deref_params_t p;

 public:
  DerefInputFn(const ImageGeometry &input, const WindowGeometry &window);
  memcpyfn_deref_params_t getParams() {return p;};
  int get_scratch_bytes();
  int get_overread_bytes();
};

int8_t *memcpyfn_deref(const memcpyfn_deref_params_t *params, int8_t *T, int8_t *X, int32_t output_v_coord,
                                 int32_t output_h_coord,
                                 int32_t output_c_coord);

struct memcpyfn_imtocol_padded_params_t {
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
};

/**
 *
 */
class ImToColPadded : public MemCpyFn {
private:
  memcpyfn_imtocol_padded_params_t p;
 public:
  ImToColPadded(const ImageGeometry &X, const WindowGeometry &K,
                              const padding_t &padding,
                              const int input_ch_per_output,
                              const int8_t pad_val);
  ImToColPadded(const Filter2dGeometry &filter_geometry, const int8_t padding_value,
           const int input_ch_per_output);
  memcpyfn_imtocol_padded_params_t getParams() {return p;};
  int get_scratch_bytes();
  int get_overread_bytes();
};

int8_t *memcpyfn_imtocol_padded(const memcpyfn_imtocol_padded_params_t *params, int8_t *T, int8_t *X, int32_t output_v_coord,
                                  int32_t output_h_coord,
                                  int32_t output_c_coord);

struct memcpyfn_imtocol_valid_params_t{
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
     * number of bytes to copy minus 32.
     */
    int32_t T_rewind;

    // The bytes to inc an X pointer by to move by one horizontal stride
    int32_t horizontal_mem_stride;

    // The bytes to inc an X pointer by to move by one vertical stride
    // and horizontally bacwards by the kernel width.
    // i.e. from X[h][w + kernel_width - 1] to X[h+1][w].
    int32_t vertical_mem_stride;

    /**
     * mask that defines how many elements are to be copied in the last channel group.
     * Should be one of 0x0000000F, 0x000000FF, ..., 0xFFFFFFFF.
     * Set to T_dontzero to 1 if the last bit must be not be zeroed
     */
     uint32_t T_vstrpv_mask;
     uint32_t T_dontzero;
};

class ImToColValid : public MemCpyFn {
  private:
  memcpyfn_imtocol_valid_params_t p;
 public:
  // input_ch_per_output lets the kernel know how many input channels to copy to
  // scratch
  ImToColValid(const ImageGeometry &X, const WindowGeometry &K,
           const int input_ch_per_output, const bool dontzero = false);
  memcpyfn_imtocol_valid_params_t getParams() {return p;};
  int get_scratch_bytes();
  int get_overread_bytes();
};

int8_t *memcpyfn_imtocol_valid(const memcpyfn_imtocol_valid_params_t *params, int8_t *T, int8_t *X, int32_t output_v_coord,
                                 int32_t output_h_coord,
                                 int32_t output_c_coord);

}  // namespace nn
#endif  // LIB_NN_MEMCPY_FN_HPP_
