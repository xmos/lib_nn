#include <cstdint>
#include <cstring>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"


struct AbstractKernelParams {
  void calculate_h_mem_stride(ImageParams &image){
    // this is only used by xformer (ideally)
    this->output_h_mem_stride = (0); // TODO:
  } 
  const int32_t h_begin, h_end;
  const int32_t w_begin, w_end;

  //The number of output channel groups that will be processed by
  //this execution
  const int32_t output_channel_group_count;


  //Used for setting the first channels slice, i.e. rather than writing to
  //slice 0-31 by offsetting it we can address slice 32 - 63, etc.
  const int32_t output_channel_slice_offset;

  //This is the number of bytes required to move from the start of a pixel 
  //(offset by output_channel_slice_offset) to the adjecent pixel to the 
  //right (offset by output_channel_slice_offset).
  const int32_t output_h_mem_stride;

  //This is the number of bytes required to move from the start of a pixel 
  //(offset by output_channel_slice_offset) on the final column of a output 
  //region to the first column of the output region pixel on the next line 
  //down(offset by output_channel_slice_offset).
  const size_t output_w_mem_stride; //different for all output regions of different widths


};

template <class T, typename TOutput, typename TInput>
class AbstractKernel {
  protected:
    AbstractKernelParams kparams;

  public:
    calc_output_pixel_slice(TOutput *Y, TInput *X, int32_t h, int32_t w);
    void execute (TOutput * Y, TInput * X) {
      Y += kparams.output_channel_slice_offset;
      for(int32_t h = kparams.h_begin; h < kparams.h_end; h++){
        for(int32_t w = kparams.w_begin; w < kparams.w_end; w++){
         static_cast<T*>(this)->calc_output_pixel_slice(Y, X, h, w);
         Y += kparams.output_w_mem_stride;
        }
        Y += kparams.output_h_mem_stride;
      }
    }
};

template <typename TOutput, typename TInput, typename TMemcopy>
class Filter2D : public AbstractKernel<Filter2D, TOutput, TInput> {
  private:
    TMemcopy * memcpy_handler;

    AggregateFn * aggregate_handler;

    OutputTransformFn * ot_handler;

    //Pointer to scratch memory
    int8_t * scratch_mem;
  public:
    void inline calc_output_pixel_slice(TOutput *Y, TInput *X, int32_t h, int32_t w) {
      
      auto * input_img = memcpy_handler->memcopy_fn(scratch_mem, X, h, w);

      for (int32_t chan_group = 0; chan_group < output_channel_group_count; chan_group++){
        vpu_ring_buffer_t A;

        aggregate_handler->aggregate_fn(&A, input_img, chan_group);

        //must calc size of current channel group
        //offset from Y in order to write out result
        //number of bytes to write to result
        //offset into transform specific arrays
        Y = ot_handler->output_transform_fn(Y, &A, chan_group);
        
      }
    };
};
