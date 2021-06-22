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

 protected:
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
  int8_t *scratch_mem;

 protected:
  /**
   * Process a single output pixel (subject to the region constraints given by
   * `kparams`
   */
  virtual void calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                       int32_t w) override;

 public:
  Filter2D(ImageGeometry &Y, ImageRegion &r, MemCpyFn *memcpy_handler,
           AggregateFn *aggregate_handler, OutputTransformFn *ot_handler,
           int8_t *scratch_mem = nullptr);

  /**
   * Construct a filter using the provided component handlers.
   */
  Filter2D(AbstractKernel::Params *kparams, MemCpyFn *memcpy_handler,
           AggregateFn *aggregate_handler, OutputTransformFn *ot_handler,
           int8_t *scratch_mem = nullptr);
};

/**
 * Base class for depthwise 2D filter kernels.
 */
class Filter2D_DW : public AbstractKernel {
 public:
  static constexpr bool UsesPerGroupMemCopy = true;

 private:
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
  int8_t *scratch_mem;

 protected:
  /**
   * Process a single output pixel (subject to the region constraints given by
   * `kparams`
   */
  virtual void calc_output_pixel_slice(int8_t *output_image,
                                       int8_t *input_image, int32_t output_row,
                                       int32_t output_col) override;

 public:
  /**
   * Construct a filter using the provided component handlers. This flavour is
   * specifically for depthwise variants.
   */
  Filter2D_DW(AbstractKernel::Params *kparams, MemCpyFn *memcpy_handler,
              AggregateFn *aggregate_handler, OutputTransformFn *ot_handler,
              int8_t *scratch_mem = nullptr,
              int output_channels_per_group = VPU_INT8_ACC_PERIOD);
};

}  // namespace nn

#endif  // LIB_NN_FILTER2D_HPP_
