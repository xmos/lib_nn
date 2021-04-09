#include <cstdint>
#include <cstring>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"
#include "../src/cpp/filt2d/geom/util.hpp"

namespace nn {
namespace filt2d {

template <class T>
class AbstractKernel {
  
public:
  class Params {
    public:

    const int32_t h_begin, h_end;
    const int32_t w_begin, w_end;

    //The number of output channel groups that will be processed by
    //this execution
    int32_t output_channel_group_count;

    //Used for setting the first channels slice, i.e. rather than writing to
    //slice 0-31 by offsetting it we can address slice 32 - 63, etc.
    int32_t output_channel_slice_offset;

    //This is the number of bytes required to move from the start of a pixel 
    //(offset by output_channel_slice_offset) to the adjecent pixel to the 
    //right (offset by output_channel_slice_offset).
    int32_t output_h_mem_stride;

    //This is the number of bytes required to move from the start of a pixel 
    //...
    int32_t output_w_mem_stride; //different for all output regions of different widths

    Params(geom::ImageGeometry &Y, ImageRegion& r):
      h_begin(r.start.row), 
      h_end(r.start.row + r.shape.height),
      w_begin(r.start.col),
      w_end(r.start.col + r.shape.width),
      output_channel_slice_offset(r.start.channel) {
      
      

      const int channels_per_group = 16; //TODO
      // output_channel_group_count = (r.channel_end - r.start.channel + channels_per_group - 1) / channels_per_group;
      output_channel_group_count = (r.shape.depth + channels_per_group - 1) / channels_per_group;

      // memory to move to the next right pixel after all current channel groups have been saved
      // i.e. this conv2d might write chs 16-31 of 68 chs, so the stride would have to be 52 channels 
      // worth of memory(enough to move from the end of the group just processed to the start of the 
      // next)
      // int output_w_mem_stride = ((Y.channels - (r.channel_end - r.channel_start)) * Y.bits_per_element ) / bits_per_byte;
      output_w_mem_stride = Y.pixelBytes();

      //memory to moved down a pixel
      // int output_h_mem_stride = (Y.width - (r.width_end - r.width_start) + r.width_start)*Y.pixelBytes();
      output_h_mem_stride = Y.rowBytes() - r.shape.width * output_w_mem_stride;




    }
  };

  protected:
  
    Params * kparams;

  public:
  AbstractKernel(Params *kparams): kparams(kparams){}
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
    Filter2D(AbstractKernel::Params * kparams, MemCpyFn * memcpy_handler, 
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem=0);

    Filter2D(geom::ImageGeometry &Y, ImageRegion& r, MemCpyFn * memcpy_handler, 
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
    Filter2D_DW(AbstractKernel::Params * kparams, MemCpyFn * memcpy_handler, 
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem);

    void calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h, int32_t w) ;
};

}
}