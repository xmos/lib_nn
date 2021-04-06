#include <cstdint>
#include <cstring>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"

struct AbstractKernelParams {
  void calculate_h_mem_stride(ImageParams &image){
    // this is only used by xformer (ideally)
    // this->output_h_mem_stride = (0); // TODO:
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
  //...
  const int32_t output_w_mem_stride; //different for all output regions of different widths

};

//region
// h_begin, h_end;
// w_begin, w_end;
// c_begin, c_end;

template <class T>
class AbstractKernel {
  protected:
  
    AbstractKernelParams * kparams;

  public:
  AbstractKernel(AbstractKernelParams *kparams): kparams(kparams){}
    //calc_output_pixel_slice(TOutput *Y, TInput *X, int32_t h, int32_t w);

  // void execute (int8_t * Y, int8_t * X) ;

  void execute (int8_t * Y, int8_t * X) {

    //dereference by h_begin and w_begin
    int bytes_per_row = kparams->output_h_mem_stride + (kparams->w_end - kparams->w_begin) * kparams->output_w_mem_stride;
    
    Y +=  kparams->h_begin * bytes_per_row + kparams->w_begin * kparams->output_w_mem_stride;

    Y += kparams->output_channel_slice_offset;

    for(int32_t h = kparams->h_begin; h < kparams->h_end; h++){
      for(int32_t w = kparams->w_begin; w < kparams->w_end; w++){
        static_cast<T*>(this)->calc_output_pixel_slice(Y, X, h, w);
        Y += kparams->output_w_mem_stride;
      }
      Y += kparams->output_h_mem_stride;
    }
  }
};

class Filter2D : public AbstractKernel<Filter2D> {
  private:
    MemCpyFn * memcpy_handler;

    AggregateFn * aggregate_handler;

    OutputTransformFn * ot_handler;

    //Pointer to scratch memory
    int8_t * scratch_mem;
  public:
    Filter2D(AbstractKernelParams * kparams, MemCpyFn * memcpy_handler, 
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem=0);

    static AbstractKernelParams make_filter2d_params(ImageParams &Y, ImageRegion& r);

    Filter2D(ImageParams &Y, ImageRegion& r, MemCpyFn * memcpy_handler, 
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem=0);

    void calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h, int32_t w) ;
};

class Filter2D_DW : public AbstractKernel<Filter2D> {
  private:
    MemCpyFn * memcpy_handler;

    AggregateFn * aggregate_handler;

    OutputTransformFn * ot_handler;

    int output_channels_per_group; //should this go in the AbstractKernelParams??

    //Pointer to scratch memory
    int8_t * scratch_mem;
  public:
    Filter2D_DW(AbstractKernelParams * kparams, MemCpyFn * memcpy_handler, 
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem);

    void calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h, int32_t w) ;
};

