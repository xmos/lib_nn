#ifndef LIB_NN_FILTER2D_HPP_
#define LIB_NN_FILTER2D_HPP_

#include "AbstractKernel.hpp"
#include "AggregateFn.hpp"
#include "MemCpyFn.hpp"
#include "OutputTransformFn.hpp"
#include "Utils.hpp"
#include "geom/util.hpp"

namespace nn {

/**
 * Base class for non-depthwise 2D filter kernels.
 *
 * This class implements `calc_output_pixel_slice()` used by `AbstractKernel`,
 * and its behavior is ultimately determined by 3 component objects supplied to
 * it, the patch handler (instance of `MemCpyFn`), the aggregation handler
 * (instance of `AggregateFn`) and the output transformer (instance of
 * `OutputTransformFn`).
 */
class Filter2D : public AbstractKernel {
 public:
  /**
   * @brief Denotes if the class uses a memcpy that copies a channel group at a
   * time, i.e. the aggregate function doesnt require all input channels to
   * compute its output (such as a depthwise conv2d), or uses a single memcpy
   * copying all input channels at once.
   */
  static constexpr bool UsesPerGroupMemCopy = false;

 public:
  /**
   * The patch handler used by this class. This determines how (and whether) a
   * region of the input image is copied into (and padded out, if necessary) a
   * scratch buffer used for im2col-like aggregation.
   */
  MemCpyFn *memcpy_handler;

  /**
   * The aggregation handler used by this class. This determines how the input
   * image's values are processed to form the 32-bit accumulator values used by
   * the output tranformer.
   */
  AggregateFn *aggregate_handler;

  /**
   * The output tranform handler used by this class. This determines how the
   * 32-bit accumulators produced by the aggregation handler are compressed down
   * into the 8-bit values which populate the output image.
   */
  OutputTransformFn *ot_handler;

  /**
   * A pointer to a scratch memory buffer. Required when im2col-like patch
   * handlers are used.
   *
   * @TODO: Where should the size of this buffer come from?
   */
  //   int8_t *scratch_mem;

 protected:
  /**
   * Process a single output pixel (subject to the region constraints given by
   * `kparams`
   */
  virtual void calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                       int32_t w, int8_t *scratch_mem,
                                       AbstractKernel::Params *kparams) override;

 public:
  Filter2D(ImageGeometry &Y, ImageRegion &r, MemCpyFn *memcpy_handler,
           AggregateFn *aggregate_handler, OutputTransformFn *ot_handler);

  /**
   * Construct a filter using the provided component handlers.
   */
  Filter2D(MemCpyFn *memcpy_handler,
           AggregateFn *aggregate_handler, OutputTransformFn *ot_handler);
};

/**
 * Base class for depthwise 2D filter kernels.
 */
class Filter2D_DW : public AbstractKernel {
 public:
  static constexpr bool UsesPerGroupMemCopy = true;

 public:
  /**
   * The patch handler used by this class. This determines how (and whether) a
   * region of the input image is copied into (and padded out, if necessary) a
   * scratch buffer used for im2col-like aggregation.
   */
  MemCpyFn *memcpy_handler;

  /**
   * The aggregation handler used by this class. This determines how the input
   * image's values are processed to form the 32-bit accumulator values used by
   * the output tranformer.
   */
  AggregateFn *aggregate_handler;

  /**
   * The output tranform handler used by this class. This determines how the
   * 32-bit accumulators produced by the aggregation handler are compressed down
   * into the 8-bit values which populate the output image.
   */
  OutputTransformFn *ot_handler;

  int output_channels_per_group;  //[asj] should this go in the
                                  // AbstractKernelParams??

  /**
   * A pointer to a scratch memory buffer. Required when im2col-like patch
   * handlers are used.
   */
  //   int8_t *scratch_mem;

  /**
   * Process a single output pixel (subject to the region constraints given by
   * `kparams`
   */
  virtual void calc_output_pixel_slice(int8_t *output_image,
                                       int8_t *input_image, int32_t output_row,
                                       int32_t output_col,
                                       int8_t *scratch_mem,
                                       AbstractKernel::Params *kparams) override;

 public:
  /**
   * Construct a filter using the provided component handlers. This flavour is
   * specifically for depthwise variants.
   */
  Filter2D_DW(MemCpyFn *memcpy_handler,
              AggregateFn *aggregate_handler, OutputTransformFn *ot_handler,
              int output_channels_per_group = VPU_INT8_ACC_PERIOD);
};

/**
 * Execute this kernel using the output image pointed to by `Y` and input
 * image pointed to by `X`, using the given filter object and kernel parameters.
 *
 * @TODO: astew: It isn't clear whether `Y` and `X` are supposed to point at
 * the base address of the output and input images, or if they're supposed to
 * point at the (first channel of the) first pixel of the images needed by
 * this filter. [update]: From looking at the output transformer
 * implementation, it looks like Y is _not_ supposed to be the image base
 * address, but instead a pointer to the first output pixel to be processed by
 * this filter. At the same time, looking at the MemCpyFn's, it looks like `X`
 * _is_ supposed to be the image base address. Doesn't this seem unnecessarily
 * confusing? Why is there an output channel offset built into kparams, but
 * not an output row or column offset?
 *
 * @param [in] Y       Pointer to the output image.
 * @param [in] X       Pointer to the input image.
 * @param [in] ak      Pointer to Filter2D object on which to operate
 * @param [in] kparams Pointer to Kernel Parameter object which identifies
 *                     what area to operate on
 * @param [in] scratch Pointer to scratch memory
 */
void execute(int8_t *Y, int8_t *X,
             AbstractKernel *ak, AbstractKernel::Params *kparams,
             int8_t *scratch = nullptr);

}  // namespace nn

#endif  // LIB_NN_FILTER2D_HPP_
