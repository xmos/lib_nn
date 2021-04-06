#include "Filter2D.hpp"
#include "vpu.hpp"

// Filter2D::Filter2D(){

// }
// template<class T>
// void AbstractKernel<T>::execute (int8_t * Y, int8_t * X) {

//   Y += kparams->output_channel_slice_offset;

//   for(int32_t h = kparams->h_begin; h < kparams->h_end; h++){
//     for(int32_t w = kparams->w_begin; w < kparams->w_end; w++){
//       static_cast<T*>(this)->calc_output_pixel_slice(Y, X, h, w);
//       Y += kparams->output_w_mem_stride;
//     }
//     Y += kparams->output_h_mem_stride;
//   }
// }
#include <iostream>
AbstractKernelParams Filter2D::make_filter2d_params(ImageParams &Y, ImageRegion& r){
  
  const int channels_per_group = 16; //TODO
  int output_channel_group_count = (r.channel_end - r.channel_start + channels_per_group - 1) / channels_per_group;

  // memory to move to the next right pixel after all current channel groups have been saved
  // i.e. this conv2d might write chs 16-31 of 68 chs, so the stride would have to be 52 channels 
  // worth of memory(enough to move from the end of the group just processed to the start of the 
  // next)
  const int bits_per_byte = 8;
  // int output_w_mem_stride = ((Y.channels - (r.channel_end - r.channel_start)) * Y.bits_per_element ) / bits_per_byte;
  int output_w_mem_stride = (Y.channels * Y.bits_per_element ) / bits_per_byte;
  assert((Y.bits_per_element % bits_per_byte) == 0);

  //memory to moved down a pixel
  int output_h_mem_stride = (Y.width - (r.width_end - r.width_start) + r.width_start)*Y.pixelBytes();

  AbstractKernelParams a = {
    .h_begin = r.height_start,
    .h_end = r.height_end,
    .w_begin = r.width_start,
    .w_end = r.width_end,
    .output_channel_slice_offset = r.channel_start,

    .output_channel_group_count = output_channel_group_count,
    .output_h_mem_stride = output_h_mem_stride,
    .output_w_mem_stride = output_w_mem_stride,
  };

  std::cout << " h_begin: " << a.h_begin <<
   " h_end: " << a.h_end<<
    " w_begin: " << a.w_begin<<
     " w_end: " << a.w_end<<
      " output_channel_slice_offset: " << a.output_channel_slice_offset<<
       " output_channel_group_count: " << a.output_channel_group_count<<
        " output_h_mem_stride: " << a.output_h_mem_stride<<
         " output_w_mem_stride: " << a.output_w_mem_stride<< std::endl;
  return a;
}


Filter2D::Filter2D(AbstractKernelParams * kparams, MemCpyFn * memcpy_handler, 
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem):
      AbstractKernel(kparams), memcpy_handler(memcpy_handler), aggregate_handler(aggregate_handler),
      ot_handler(ot_handler), scratch_mem(scratch_mem)
{

}

/*
  This is going to compute the output for output_channel_group_count channel groups of
  the output. The pointer is going to be set to the begining on the next output by 
  output_w_mem_stride. This allows it to address sub-channel regions.
*/
void Filter2D::calc_output_pixel_slice(int8_t * Y, int8_t * X, int32_t h, int32_t w){
      
  int8_t * input_img = memcpy_handler->memcopy_fn(scratch_mem, X, h, w); //copy all input channels, channel start is implicitly 0.

  std::cout << "h: " << h << " w:"<<w <<std::endl;
  for (int32_t chan_group = 0; chan_group < kparams->output_channel_group_count; chan_group++){
    vpu_ring_buffer_t A;

    aggregate_handler->aggregate_fn(&A, input_img, chan_group);

    Y = ot_handler->output_transform_fn(Y, &A, chan_group);
    
  }
}

//This is an example of a depthwise conv or max pool
void Filter2D_DW::calc_output_pixel_slice(int8_t * Y, int8_t * X, int32_t h, int32_t w){
      
  for (int32_t chan_group = 0; chan_group < kparams->output_channel_group_count; chan_group++){

    vpu_ring_buffer_t A;

    int c = kparams->output_channel_slice_offset + chan_group * output_channels_per_group;

    int8_t * input_img = memcpy_handler->memcopy_fn(scratch_mem, X, h, w, c); //will know how many channels it is copying

    aggregate_handler->aggregate_fn(&A, input_img, chan_group);

    //must calc size of current channel group
    //offset from Y in order to write out result
    //number of bytes to write to result
    //offset into transform specific arrays
    Y = ot_handler->output_transform_fn(Y, &A, chan_group);
    
  }
}
