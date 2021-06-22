#ifndef LIB_NN_ABSTRACT_KERNEL_HPP_
#define LIB_NN_ABSTRACT_KERNEL_HPP_

#include "geom/Filter2dGeometry.hpp"

namespace nn {

/**
 * Class representing an executable filter kernel. A concrete instance of this
 * class processes a particular region of the output image associated with the
 * filter.
 */
class AbstractKernel {
 public:
  /**
   * Struct indicating a (3D) rectangular sub-region of an image, as well as how
   * to iterate through pixels within that region of the image.
   *
   * @see AbstractKernel
   */
  class Params {
   public:
    /**
     * The first (`h_begin`; inclusive) and final (`h_end`; exclusive) rows of
     * the output image to be processed when the corresponding filter is
     * executed.
     */
    const int32_t h_begin, h_end;

    /**
     * The first (`w_begin`; inclusive) and final (`w_end`; exclusive) columns
     * of the output image to be processed when the corresponding filter is
     * executed.
     */
    const int32_t w_begin, w_end;

    /**
     * The number of output channel groups that will be processed when this
     * filter is executed.
     */
    int32_t output_channel_group_count;

    /**
     * The first output channel to be processed when this filter is executed.
     *
     * This filter will process output channels in the range:
     *  [`output_channel_slice_offset`, `output_channel_slice_offset` +
     * `output_channel_group_count` * `channels_per_output_group`)
     *
     * Used for setting the first channels slice, i.e. rather than writing to
     * slice 0-31 by offsetting it we can address slice 32 - 63, etc.
     *
     * @TODO: astew: Why not just have c_begin and c_end like with the rows and
     * columns handled by this filter?
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
     * to the adjecent pixel to the right.
     */
    int32_t output_w_mem_stride;

    Params(const ImageGeometry &output_image, const ImageRegion &output_region,
           const int channels_per_group)
        : h_begin(output_region.start.row),
          h_end(output_region.EndVect().row),
          w_begin(output_region.start.col),
          w_end(output_region.EndVect().col),
          output_channel_group_count(
              (output_region.shape.depth + channels_per_group - 1) /
              channels_per_group),
          output_channel_slice_offset(output_region.start.channel),
          output_h_mem_stride(
              output_image.GetStride(1, -output_region.shape.width, 0)),
          output_w_mem_stride(output_image.GetStride(0, 1, 0)) {}
  };

 protected:
  /**
   * Parameters describing the region of the output image to be processed by
   * this filter, as well as how to iterate over the region.
   */
  Params *kparams;

  virtual void calc_output_pixel_slice(int8_t *output_image,
                                       int8_t *input_image, int32_t output_row,
                                       int32_t output_col) = 0;

 public:
  /**
   * Constructor.
   *
   * @param [in] kparams  Parameters describing the output region to be
   * processed.
   */
  AbstractKernel(Params *kparams) : kparams(kparams) {}

  /**
   * Execute this kernel using the output image pointed to by `Y` and input
   * image pointed to by `X`.
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
   * @param [in] Y  Pointer to the output image.
   * @param [in] X  Pointer to the input image.
   */
  void execute(int8_t *Y, int8_t *X) {
    int bytes_per_row =
        kparams->output_h_mem_stride +
        (kparams->w_end - kparams->w_begin) * kparams->output_w_mem_stride;

    Y += kparams->h_begin * bytes_per_row +
         kparams->w_begin * kparams->output_w_mem_stride;

    Y += kparams->output_channel_slice_offset;

    for (int32_t h = kparams->h_begin; h < kparams->h_end; h++) {
      for (int32_t w = kparams->w_begin; w < kparams->w_end; w++) {
        this->calc_output_pixel_slice(Y, X, h, w);
        Y += kparams->output_w_mem_stride;
      }
      Y += kparams->output_h_mem_stride;
    }
  }
};

}  // namespace nn

#endif  // LIB_NN_ABSTRACT_KERNEL_HPP_
