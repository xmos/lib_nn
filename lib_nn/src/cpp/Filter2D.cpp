#include "AbstractKernel.hpp"
#include "vpu.hpp"

using namespace nn;

void dconv_calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                          int32_t w, int8_t *scratch_mem,
                                          abstract_kernel_params_t *kparams, conv_params_t *p, int8_t* weights, int16_t* muls_and_biases) {
  // We could move this parameter out if we want to configure it
  const int output_channels_per_group = VPU_INT8_ACC_PERIOD;
  
  const auto output_groups = kparams->output_channel_group_count;

  for (int32_t chan_group = 0; chan_group < output_groups; chan_group++) {
    VPURingBuffer A;

    int c = kparams->output_channel_slice_offset +
            chan_group * output_channels_per_group;

    // This will know how many channels it is copying
    int8_t *input_img =
        p->memcopy_fn(p->mem_p, scratch_mem, X, h, w, c);

    p->aggregate_fn(p->agg_p, &A, input_img, chan_group, weights);

    // must calc size of current channel group
    // offset from Y in order to write out result
    // number of bytes to write to result
    // offset into transform specific arrays
    Y = p->output_transform_fn(p->ot_p, Y, &A, chan_group, muls_and_biases);
  }
}

void conv_calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h,
                                       int32_t w, int8_t *scratch_mem,
                                       abstract_kernel_params_t *kparams, conv_params_t *p, int8_t* weights, int16_t* muls_and_biases) {
  int8_t *input_img = p->memcopy_fn(p->mem_p,
      scratch_mem, X, h, w, 0);  // copy all input channels, channel start is implicitly 0.
  for (int32_t output_chan_group = 0;
       output_chan_group < kparams->output_channel_group_count;
       ++output_chan_group) {
    VPURingBuffer A;
    p->aggregate_fn(p->agg_p, &A, input_img, output_chan_group, weights);

    Y = p->output_transform_fn(p->ot_p, Y, &A, output_chan_group, muls_and_biases);
  }
}

void nn::execute(int8_t *Y, int8_t *X, conv_params_t *ak,
             abstract_kernel_params_t *kparams, int8_t* weights, int16_t* muls_and_biases, bool isConv, int8_t *scratch) {
  int bytes_per_row =
      kparams->output_h_mem_stride +
      (kparams->w_end - kparams->w_begin) * kparams->output_w_mem_stride;

  Y += kparams->h_begin * bytes_per_row +
       kparams->w_begin * kparams->output_w_mem_stride;

  Y += kparams->output_channel_slice_offset;
  if(isConv){
    for (int32_t h = kparams->h_begin; h < kparams->h_end; h++) {
      for (int32_t w = kparams->w_begin; w < kparams->w_end; w++) {
        conv_calc_output_pixel_slice(Y, X, h, w, scratch, kparams, ak, weights, muls_and_biases);
        Y += kparams->output_w_mem_stride;
      }
      Y += kparams->output_h_mem_stride;
    }
  } else {
    for (int32_t h = kparams->h_begin; h < kparams->h_end; h++) {
      for (int32_t w = kparams->w_begin; w < kparams->w_end; w++) {
        dconv_calc_output_pixel_slice(Y, X, h, w, scratch, kparams, ak, weights, muls_and_biases);
        Y += kparams->output_w_mem_stride;
      }
      Y += kparams->output_h_mem_stride;
    }
  }
}