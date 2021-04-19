#include <cstdint>
#include <cstring>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"
#include "../src/cpp/filt2d/geom/util.hpp"

#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"

namespace nn {

  /**
   * Class representing an executable filter kernel. A concrete instance of this class processes a
   * particular region of the output image associated with the filter.
   */
  template <class T>
  class AbstractKernel {
    
  public:
    /**
     * Struct indicating a (3D) rectangular sub-region of an image, as well as how to iterate through
     * pixels within that region of the image.
     * 
     * @see AbstractKernel
     */
    class Params {
      public:

  /**
     * The first (`h_begin`; inclusive) and final (`h_end`; exclusive) rows of the output image
     * to be processed when the corresponding filter is executed.
     */
      const int32_t h_begin, h_end;

    /**
     * The first (`w_begin`; inclusive) and final (`w_end`; exclusive) columns of the output image
     * to be processed when the corresponding filter is executed.
     */
      const int32_t w_begin, w_end;

    /**
     * The number of output channel groups that will be processed when this filter is executed.
     */
      int32_t output_channel_group_count;

  //Used for setting the first channels slice, i.e. rather than writing to
    //slice 0-31 by offsetting it we can address slice 32 - 63, etc.

    /**
     * The first output channel to be processed when this filter is executed.
     * 
     * This filter will process output channels in the range:
     *  [`output_channel_slice_offset`, `output_channel_slice_offset` + `output_channel_group_count` * `channels_per_output_group`)
     * 
     * @TODO: astew: Why not just have c_begin and c_end like with the rows and columns handled by this filter?
     */
      int32_t output_channel_slice_offset;

      /**
     * This is the number of bytes required to move from the start of a pixel 
     * (offset by output_channel_slice_offset) on the final column of a output 
     * region to the first column of the output region pixel on the next line 
     * down(offset by output_channel_slice_offset).
     */
      int32_t output_h_mem_stride;

      /**
     * This is the number of bytes required to move from the start of a pixel 
     * (offset by output_channel_slice_offset) to the adjecent pixel to the 
     * right (offset by output_channel_slice_offset).
     */
      int32_t output_w_mem_stride; //different for all output regions of different widths

      Params(ImageGeometry &Y, ImageRegion& r):
        h_begin(r.start.row), 
        h_end(r.start.row + r.shape.height),
        w_begin(r.start.col),
        w_end(r.start.col + r.shape.width),
        output_channel_slice_offset(r.start.channel) {
        
        

        const int channels_per_group = 16; //TODO
        // output_channel_group_count = (r.channel_end - r.start.channel + channels_per_group - 1) / channels_per_group;
        output_channel_group_count = (r.shape.depth + channels_per_group - 1) / channels_per_group;

        // memory to move to the next right pixel after all current channel groups have been saved
        // i.e. this conv2d might write chs 16-31 of 68 chs, so the stride would have to be 52 channels 
        // worth of memory(enough to move from the end of the group just processed to the start of the 
        // next)
        // int output_w_mem_stride = ((Y.channels - (r.channel_end - r.channel_start)) * Y.bits_per_element ) / bits_per_byte;
        output_w_mem_stride = Y.pixelBytes();

        //memory to moved down a pixel
        // int output_h_mem_stride = (Y.width - (r.width_end - r.width_start) + r.width_start)*Y.pixelBytes();
        output_h_mem_stride = Y.rowBytes() - r.shape.width * output_w_mem_stride;

      }

      Params(const ImageGeometry& image, 
            const ImageRegion& region, 
            const int channels_per_group = VPU_INT8_ACC_PERIOD)
        : h_begin(region.start.row), h_end(region.endVect().row),
          w_begin(region.start.col), w_end(region.endVect().col),
          output_channel_slice_offset(region.start.channel) 
      {
        this->output_channel_group_count = (region.shape.depth + channels_per_group - 1) / channels_per_group;
        
        // memory to move to the next right pixel after all current channel groups have been saved
        // i.e. this conv2d might write chs 16-31 of 68 chs, so the stride would have to be 52 channels 
        // worth of memory(enough to move from the end of the group just processed to the start of the 
        // next)
        this->output_w_mem_stride = ( image.depth * image.channel_depth );

        //memory to moved down a pixel
        this->output_h_mem_stride = image.rowBytes() - region.shape.width * this->output_w_mem_stride;
      }
    };

    protected:

      /**
       * Parameters describing the region of the output image to be processed by this filter, 
       * as well as how to iterate over the region.
       */
      Params * kparams;

    public:

      /**
       * Constructor.
       * 
       * @param [in] kparams  Parameters describing the output region to be processed.
       */
    AbstractKernel(Params *kparams): kparams(kparams){}
      //calc_output_pixel_slice(TOutput *Y, TInput *X, int32_t h, int32_t w);

    // void execute (int8_t * Y, int8_t * X) ;

      /**
       * Execute this kernel using the output image pointed to by `Y` and input image pointed to by `X`.
       * 
       * @TODO: astew: It isn't clear whether `Y` and `X` are supposed to point at the base address of the
       *        output and input images, or if they're supposed to point at the (first channel of the) first
       *        pixel of the images needed by this filter.
       *        [update]: From looking at the output transformer implementation, it looks like Y is _not_
       *         supposed to be the image base address, but instead a pointer to the first output pixel to
       *         be processed by this filter.
       *         At the same time, looking at the MemCpyFn's, it looks like `X` _is_ supposed to be the image
       *         base address. Doesn't this seem unnecessarily confusing? Why is there an output channel offset 
       *         built into kparams, but not an output row or column offset?
       * 
       * @param [in] Y  Pointer to the output image.
       * @param [in] X  Pointer to the input image.
       */
    void execute (int8_t * Y, int8_t * X) {

      //dereference by h_begin and w_begin
      int bytes_per_row = kparams->output_h_mem_stride + (kparams->w_end - kparams->w_begin) * kparams->output_w_mem_stride;
      
      Y +=  kparams->h_begin * bytes_per_row + kparams->w_begin * kparams->output_w_mem_stride;

      Y += kparams->output_channel_slice_offset;

      for(int32_t h = kparams->h_begin; h < kparams->h_end; h++){
        for(int32_t w = kparams->w_begin; w < kparams->w_end; w++){
          static_cast<T*>(this)->calc_output_pixel_slice(Y, X, h, w);
          Y += kparams->output_w_mem_stride;
        }
        Y += kparams->output_h_mem_stride;
      }
    }
  };

  /**
   * Base class for non-depthwise 2D filter kernels.
   * 
   * This class implements `calc_output_pixel_slice()` used by `AbstractKernel`, and its behavior is
   * ultimately determined by 3 component objects supplied to it, the patch handler (instance of `MemCpyFn`),
   * the aggregation handler (instance of `AggregateFn`) and the output transformer (instance of `OutputTransformFn`).
   */
  class Filter2D : public AbstractKernel<Filter2D> {
    public:
      static constexpr bool UsesPerGroupMemCopy = false;

    private:

      /**
       * The patch handler used by this class. This determines how (and whether) a region of the input image is 
       * copied into (and padded out, if necessary) a scratch buffer used for im2col-like aggregation.
       */
      MemCpyFn * memcpy_handler;

      /**
       * The aggregation handler used by this class. This determines how the input image's values are processed
       * to form the 32-bit accumulator values used by the output tranformer.
       */
      AggregateFn * aggregate_handler;

      /**
       * The output tranform handler used by this class. This determines how the 32-bit accumulators produced by
       * the aggregation handler are compressed down into the 8-bit values which populate the output image.
       */
      OutputTransformFn * ot_handler;

      /**
       * A pointer to a scratch memory buffer. Required when im2col-like patch handlers are used.
       * 
       * @TODO: Where should the size of this buffer come from?
       */
      int8_t * scratch_mem;
      
    protected:

      /**
       * Process a single output pixel (subject to the region constraints given by `kparams`
       */
      void calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h, int32_t w) ;

    public:

      Filter2D(ImageGeometry &Y, ImageRegion& r, MemCpyFn * memcpy_handler, 
        AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem=0);
        
      /**
       * Construct a filter using the provided component handlers.
       */
      Filter2D(AbstractKernel::Params * kparams, 
              MemCpyFn * memcpy_handler, 
              AggregateFn * aggregate_handler, 
              OutputTransformFn * ot_handler, 
              int8_t * scratch_mem=nullptr);

      // Because AbstractKernel calls calc_output_pixel_slice(), which is protected (and not a virtual function
      // of AbstractKernel), AbstractKernel<Filter2D> must be declared a friend class.
      friend class AbstractKernel<Filter2D>;

  };

  /**
   * Base class for depthwise 2D filter kernels.
   */
  class Filter2D_DW : public AbstractKernel<Filter2D_DW> {
    public:
      static constexpr bool UsesPerGroupMemCopy = true;

    private:

      /**
       * The patch handler used by this class. This determines how (and whether) a region of the input image is 
       * copied into (and padded out, if necessary) a scratch buffer used for im2col-like aggregation.
       */
      MemCpyFn * memcpy_handler;

      /**
       * The aggregation handler used by this class. This determines how the input image's values are processed
       * to form the 32-bit accumulator values used by the output tranformer.
       */
      AggregateFn * aggregate_handler;

      /**
       * The output tranform handler used by this class. This determines how the 32-bit accumulators produced by
       * the aggregation handler are compressed down into the 8-bit values which populate the output image.
       */
      OutputTransformFn * ot_handler;

      int output_channels_per_group; //should this go in the AbstractKernelParams??

      /**
       * A pointer to a scratch memory buffer. Required when im2col-like patch handlers are used.
       * 
       * @TODO: Where should the size of this buffer come from?
       */
      int8_t * scratch_mem;

    protected:

      /**
       * Process a single output pixel (subject to the region constraints given by `kparams`
       */
      void calc_output_pixel_slice(int8_t *Y, 
                                  int8_t *X, 
                                  int32_t h, 
                                  int32_t w) ;

    public:

      /**
       * Construct a filter using the provided component handlers.
       */
      Filter2D_DW(AbstractKernel::Params * kparams, 
                  MemCpyFn * memcpy_handler, 
                  AggregateFn * aggregate_handler, 
                  OutputTransformFn * ot_handler, 
                  int8_t * scratch_mem = nullptr,
                  int output_channels_per_group = VPU_INT8_ACC_PERIOD);

        
      // Because AbstractKernel calls calc_output_pixel_slice(), which is protected (and not a virtual function
      // of AbstractKernel), AbstractKernel<Filter2D> must be declared a friend class.
      friend class AbstractKernel<Filter2D_DW>;

  };

}