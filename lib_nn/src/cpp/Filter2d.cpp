#include "Filter2D.hpp"

#include "vpu.hpp"

using namespace nn;

constexpr bool Filter2D::UsesPerGroupMemCopy;
constexpr bool Filter2D_DW::UsesPerGroupMemCopy;

Filter2D::Filter2D(AbstractKernel::Params *kparams, MemCpyFn *memcpy_handler,
                   AggregateFn *aggregate_handler,
                   OutputTransformFn *ot_handler, int8_t *scratch_mem)
    : AbstractKernel(kparams),
      memcpy_handler(memcpy_handler),
      aggregate_handler(aggregate_handler),
      ot_handler(ot_handler),
      scratch_mem(scratch_mem) {}

/*
  This is going to compute the output for output_channel_group_count channel
  groups of the output. The pointer is going to be set to the begining on the
  next output by output_w_mem_stride. This allows it to address sub-channel
  regions.
*/
void Filter2D::calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                       int32_t w) {
  int8_t *input_img = memcpy_handler->memcopy_fn(
      scratch_mem, X, h,
      w);  // copy all input channels, channel start is implicitly 0.

  for (int32_t chan_group = 0; chan_group < kparams->output_channel_group_count;
       chan_group++) {
    VPURingBuffer A;

    aggregate_handler->aggregate_fn(&A, input_img, chan_group);

    Y = ot_handler->output_transform_fn(Y, &A, chan_group);
  }
}

Filter2D_DW::Filter2D_DW(AbstractKernel::Params *kparams,
                         MemCpyFn *memcpy_handler,
                         AggregateFn *aggregate_handler,
                         OutputTransformFn *ot_handler, int8_t *scratch_mem,
                         int output_channels_per_group)
    : AbstractKernel(kparams),
      memcpy_handler(memcpy_handler),
      aggregate_handler(aggregate_handler),
      ot_handler(ot_handler),
      output_channels_per_group(output_channels_per_group),
      scratch_mem(scratch_mem) {}

// This is an example of a depthwise conv or max pool
void Filter2D_DW::calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                          int32_t w) {
  const auto output_groups = this->kparams->output_channel_group_count;

  for (int32_t chan_group = 0; chan_group < output_groups; chan_group++) {
    VPURingBuffer A;

    int c = this->kparams->output_channel_slice_offset +
            chan_group * this->output_channels_per_group;

    // This will know how many channels it is copying
    int8_t *input_img =
        this->memcpy_handler->memcopy_fn(this->scratch_mem, X, h, w, c);

    this->aggregate_handler->aggregate_fn(&A, input_img, chan_group);

    // must calc size of current channel group
    // offset from Y in order to write out result
    // number of bytes to write to result
    // offset into transform specific arrays
    Y = this->ot_handler->output_transform_fn(Y, &A, chan_group);
  }
}
