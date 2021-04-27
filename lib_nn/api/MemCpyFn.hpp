#include <cstdint>
#include <cstring>

#include "geom/Filter2dGeometry.hpp"

namespace nn
{

  /**
   * Interface implemented by all patch handlers.
   */
  class MemCpyFn
  {
  public:
    /**
       * Copy the relevant region of the input image to the patch buffer, and return a pointer
       * to where the aggregation handler should begin consuming inputs.
       * 
       * @TODO: astew: The `c` below is confusing, since `h` and `w` are in the output image's
       *               coordinate space, one is tempted to thing `c` also is, but `c` is actually
       *               in the input image's coordinate space. e.g. for non-depthwise mem copies
       *               `c` will always be zero (because we copy all input channels), and for depthwise
       *               mem copies while it's the output channel coordinate that is given, that only
       *               works because in a depthwise op the output and input channel indices are the
       *               same.
       *               I suggest that memcpy_fn() should always be given the current output channel
       *               (first of the current output channel group), and internally any MemCpyFn implementation
       *               that is supposed to work for both depthwise and non-depthwise ops should just
       *               keep a 'channel stride' parameter (@see nn::WindowGeometry) that will be either 0
       *               (for dense convolutions) or 1 (for depthwise convolutions). 
       *               This approach will also generalize if we ever end up with convolutions in which, for example,
       *               output channel 0 is a function of both input channels 0 and 1 (but no others), and output
       *               channel 1 is a function of both input channels 2 and 3 (but no others), etc, by using a
       *               channel stride value of 2.
       * 
       * @param [inout] T   pointer to patch buffer
       * @param [in] X      pointer to base address of the input image
       * @param [in] h      row index of the _OUTPUT_ pixel being processed 
       * @param [in] w      column index of the _OUTPUT_ pixel being processed
       * @param [in] c      channel index of the _INPUT_ image from which to start copying elements
       */
    virtual int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c = 0) = 0;

      /**
       * 
       */
    virtual size_t get_scratch_bytes() = 0;

      /**
       * 
       */
    virtual size_t get_overread_bytes() = 0;
  };

  /**
   * 
   */
  class DerefInputFn : public MemCpyFn
  {

  public:
    class Params
    {

    public:
      int32_t bytes_per_h_line;
      int32_t bytes_per_pixel;

      Params(int32_t bytes_per_h_line, int32_t bytes_per_pixel);

      Params(const ImageGeometry &X, const WindowGeometry &K);

      Params(const Filter2dGeometry &filter_geometry);

      Params(std::istream &stream);

      void Serialize(std::ostream &stream) const;
    };

    Params *params;

  public:
    DerefInputFn(Params *params) : params(params){};
    int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c);
    size_t get_scratch_bytes();
    size_t get_overread_bytes();
  };

  /**
   * 
   */
  class ImToColPadded : public MemCpyFn
  {
  public:
    class Params
    {

    public:
      int32_t kernel_height;
      int32_t kernel_width;

      int32_t vertical_stride;
      int32_t horizontal_stride;
      int32_t vertical_dilation;
      int32_t horizontal_dilation;

      padding_t padding;

      int32_t input_v_length;
      int32_t input_h_length;

      int32_t padding_val;

      int32_t bytes_per_h_line;
      int32_t bytes_per_pixel;

      int32_t horizontal_mem_stride;

      int32_t bytes_per_copy_per_channel;

    public:
      Params(const ImageGeometry &X,
             const WindowGeometry &K,
             const padding_t &padding,
             const int input_ch_per_output,
             const int8_t pad_val);

        Params(const Filter2dGeometry& filter_geometry,
               const int8_t padding_value,
               const int channels_per_output);

      Params(std::istream &stream);

      void Serialize(std::ostream &stream) const;
    };

    Params *params;

  public:
    ImToColPadded(Params *p) : params(p) {}
    int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c);
    size_t get_scratch_bytes();
    size_t get_overread_bytes();
  };

  /**
   * 
   */
  class ImToColValid : public MemCpyFn
  {
  public:
    struct Params
    {
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
     * Note that this should be rounded up if the number of words is not integral.
     */
      int32_t input_channel_groups; 


    /**
     * The difference between the number of bytes actually copied and the target number of bytes to copy.
     */
      int32_t T_rewind;

      // The bytes to inc an X pointer by to move by one horizontal stride
      int32_t horizontal_mem_stride;

      // The bytes to inc an X pointer by to move by one vertical stride
      // and horizontally bacwards by the kernel width.
      // i.e. from X[h][w + kernel_width - 1] to X[h+1][w].
      int32_t vertical_mem_stride;

      Params(const ImageGeometry &X, const WindowGeometry &K, const int input_ch_per_output);
    };

    Params *params;

  public:
    //input_ch_per_output lets the kernel know how many input channels to copy to scratch
    ImToColValid(Params *params) : params(params){};

    size_t get_scratch_bytes();
    size_t get_overread_bytes();

    int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c);
  };
}
