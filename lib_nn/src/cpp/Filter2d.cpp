#include "Filter2D.hpp"
#include "vpu.hpp"

// Filter2D::Filter2D(){

// }
template<class T>
void AbstractKernel<T>::execute (int8_t * Y, int8_t * X) {
  Y += kparams.output_channel_slice_offset;
  for(int32_t h = kparams.h_begin; h < kparams.h_end; h++){
    for(int32_t w = kparams.w_begin; w < kparams.w_end; w++){
      static_cast<T*>(this)->calc_output_pixel_slice(Y, X, h, w);
      Y += kparams.output_w_mem_stride;
    }
    Y += kparams.output_h_mem_stride;
  }
}

void Filter2D::calc_output_pixel_slice(int8_t * Y, int8_t * X, int32_t h, int32_t w){
      
  int8_t * input_img = memcpy_handler->memcopy_fn(scratch_mem, X, h, w);

  for (int32_t chan_group = 0; chan_group < kparams.output_channel_group_count; chan_group++){
    vpu_ring_buffer_t A;

    aggregate_handler->aggregate_fn(&A, input_img, chan_group);

    //must calc size of current channel group
    //offset from Y in order to write out result
    //number of bytes to write to result
    //offset into transform specific arrays
    Y = ot_handler->output_transform_fn(Y, &A, chan_group);
    
  }
}

// void Filter2D::execute(int8_t * Y, int8_t * X){

//   Y += kparams.output_channel_slice_offset;
//   for(int32_t h = kparams.h_begin; h < kparams.h_end; h++){
//     for(int32_t w = kparams.w_begin; w < kparams.w_end; w++){
//   ;
//     }
//     Y += kparams.output_h_mem_stride;
//   }
// }