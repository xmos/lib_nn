#ifndef LIB_NN_ABSTRACT_KERNEL_HPP_
#define LIB_NN_ABSTRACT_KERNEL_HPP_

#include "vpu.hpp"
#include "geom/Filter2dGeometry.hpp"

namespace nn {

struct abstract_kernel_params_t {
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
     * The first of output channel groups that will be processed when this
     * filter is executed.
     */
    int32_t output_channel_group_begin;

    /**
     * The last of output channel groups that will be processed when this
     * filter is executed.
     */
    int32_t output_channel_group_end;

    /**
     * The first output channel to be processed when this filter is executed.
     *
     * This filter will process output channels in the range:
     *  [`output_channel_slice_offset`, `output_channel_slice_offset` +
     * (`output_channel_group_end` - `output_channel_group_begin`) * `channels_per_output_group`)
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

    /**
     * This is the number of bytes added to the input pointer when we chain
     * convolutions.
     */
    int32_t input_offset;
};

/**
 * Class representing an executable filter kernel. A concrete instance of this
 * class processes a particular region of the output image associated with the
 * filter.
 */
class AbstractKernel {
  private:
  abstract_kernel_params_t p;
 public:
  /**
   * Constructor.
   */
  AbstractKernel(const ImageGeometry &output_image, const ImageRegion &output_region,
           const int channels_per_output_group) : 
           AbstractKernel (output_image, output_region, channels_per_output_group, 0, 0, 1, 1, 0)
          {};
  
  AbstractKernel(const ImageGeometry &output_image, const ImageRegion &output_region,
           const int channels_per_output_group, 
           const int sub_h, const int sub_w, 
           const int stride_h, const int stride_w, const int input_offset) :
          p{(output_region.start.row  - sub_h )/ stride_h ,
          (output_region.EndVect().row - sub_h + stride_h - 1) /stride_h ,
          (output_region.start.col - sub_w )/ stride_w,
          (output_region.EndVect().col - sub_w + stride_w-1) / stride_w ,
          (output_region.start.channel / channels_per_output_group),
          (output_region.start.channel / channels_per_output_group) + ((output_region.shape.depth + channels_per_output_group - 1) /
          channels_per_output_group),
          output_region.start.channel + output_image.GetStride(sub_h, sub_w, 0), 
          output_image.GetStride(stride_h, -((output_region.shape.width - sub_w + stride_w - 1) / stride_w)*stride_w, 0),
          output_image.GetStride(0, stride_w, 0),
          input_offset} {};

  abstract_kernel_params_t getParams() {return p;};

};

typedef int8_t* (*MemFnType)(const void*, int8_t*, int8_t*, int32_t, int32_t, int32_t);
typedef void (*AggFnType)(const void *, VPURingBuffer*, int8_t*, int32_t, int8_t*);
typedef int8_t* (*OtFnType)(const void*, int8_t *, VPURingBuffer*, int32_t, int16_t*);

enum conv_type {
  CONV, DCONV, I16CONV
};

struct conv_params_t{
    void *mem_p;
    void *agg_p;
    void *ot_p;
    MemFnType memcopy_fn;
    AggFnType aggregate_fn;
    OtFnType output_transform_fn;
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
void execute(int8_t *Y, int8_t *X, conv_params_t *ak,
             abstract_kernel_params_t *kparams, int8_t* weights, int16_t* muls_and_biases, conv_type c_type, int8_t *scratch = nullptr);


}  // namespace nn

#endif  // LIB_NN_ABSTRACT_KERNEL_HPP_
