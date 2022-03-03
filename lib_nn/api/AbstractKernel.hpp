#ifndef LIB_NN_ABSTRACT_KERNEL_HPP_
#define LIB_NN_ABSTRACT_KERNEL_HPP_

#include "Serialisable.hpp"
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
  class Params : public Serialisable {
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
           const int channels_per_output_group)
        : h_begin(output_region.start.row),
          h_end(output_region.EndVect().row),
          w_begin(output_region.start.col),
          w_end(output_region.EndVect().col),
          output_channel_group_count(
              (output_region.shape.depth + channels_per_output_group - 1) /
              channels_per_output_group),
          output_channel_slice_offset(output_region.start.channel),
          output_h_mem_stride(
              output_image.GetStride(1, -output_region.shape.width, 0)),
          output_w_mem_stride(output_image.GetStride(0, 1, 0)) {}
  };

 public:
  /**
   * Parameters describing the region of the output image to be processed by
   * this filter, as well as how to iterate over the region.
   */

  virtual void calc_output_pixel_slice(int8_t *output_image,
                                       int8_t *input_image, int32_t output_row,
                                       int32_t output_col, int8_t *scratch,
                                       AbstractKernel::Params *kparams) = 0;

  /**
   * Constructor.
   *
   * @param [in] kparams  Parameters describing the output region to be
   * processed.
   */
  AbstractKernel() {}
};

}  // namespace nn

#endif  // LIB_NN_ABSTRACT_KERNEL_HPP_
