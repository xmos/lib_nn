#include "Filter2D.hpp"

#include "vpu.hpp"

using namespace nn;

constexpr bool Filter2D::UsesPerGroupMemCopy;
constexpr bool Filter2D_DW::UsesPerGroupMemCopy;

Filter2D::Filter2D(MemCpyFn *memcpy_handler,
                   AggregateFn *aggregate_handler,
                   OutputTransformFn *ot_handler)
    : AbstractKernel(),
      memcpy_handler(memcpy_handler),
      aggregate_handler(aggregate_handler),
      ot_handler(ot_handler) {}

/*
  This is going to compute the output for output_channel_group_count channel
  groups of the output. The pointer is going to be set to the begining on the
  next output by output_w_mem_stride. This allows it to address sub-channel
  regions.
*/
void Filter2D::calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                       int32_t w, int8_t *scratch_mem,
                                       AbstractKernel::Params *kparams) {
  int8_t *input_img = memcpy_handler->memcopy_fn(
      scratch_mem, X, h,
      w);  // copy all input channels, channel start is implicitly 0.
  for (int32_t output_chan_group = 0;
       output_chan_group < kparams->output_channel_group_count;
       ++output_chan_group) {
    VPURingBuffer A;
    aggregate_handler->aggregate_fn(&A, input_img, output_chan_group);

    Y = ot_handler->output_transform_fn(Y, &A, output_chan_group);
  }
}

Filter2D_DW::Filter2D_DW(MemCpyFn *memcpy_handler,
                         AggregateFn *aggregate_handler,
                         OutputTransformFn *ot_handler,
                         int output_channels_per_group)
    : AbstractKernel(),
      memcpy_handler(memcpy_handler),
      aggregate_handler(aggregate_handler),
      ot_handler(ot_handler),
      output_channels_per_group(output_channels_per_group) {}

// This is an example of a depthwise conv or max pool
void Filter2D_DW::calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                          int32_t w, int8_t *scratch_mem,
                                          AbstractKernel::Params *kparams) {
  const auto output_groups = kparams->output_channel_group_count;

  for (int32_t chan_group = 0; chan_group < output_groups; chan_group++) {
    VPURingBuffer A;

    int c = kparams->output_channel_slice_offset +
            chan_group * this->output_channels_per_group;

    // This will know how many channels it is copying
    int8_t *input_img =
        this->memcpy_handler->memcopy_fn(scratch_mem, X, h, w, c);

    this->aggregate_handler->aggregate_fn(&A, input_img, chan_group);

    // must calc size of current channel group
    // offset from Y in order to write out result
    // number of bytes to write to result
    // offset into transform specific arrays
    Y = this->ot_handler->output_transform_fn(Y, &A, chan_group);
  }
}

void nn::execute(int8_t *Y, int8_t *X,
                AbstractKernel *ak, AbstractKernel::Params *kparams,
                int8_t *scratch) {
  int bytes_per_row =
      kparams->output_h_mem_stride +
      (kparams->w_end - kparams->w_begin) * kparams->output_w_mem_stride;

  Y += kparams->h_begin * bytes_per_row +
       kparams->w_begin * kparams->output_w_mem_stride;

  Y += kparams->output_channel_slice_offset;
  for (int32_t h = kparams->h_begin; h < kparams->h_end; h++) {
    for (int32_t w = kparams->w_begin; w < kparams->w_end; w++) {
        ak->calc_output_pixel_slice(Y, X, h, w, scratch, kparams);
      Y += kparams->output_w_mem_stride;
    }
    Y += kparams->output_h_mem_stride;
  }
}
