#include <cstdint>
#include <cstring>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"



class Filter2D  {

  const int32_t h_begin, h_end;
  const int32_t w_begin, w_end;

  //The number of output channel groups that will be processed by
  //this execution
  const int32_t output_channel_group_count;

  //Used for setting the first channels slice, i.e. rather than writing to
  //slice 0-31 by offsetting it we can address slice 32 - 63, etc.
  const size_t output_channel_slice_offset;

  //These describe how to move around in the output memory space.

  //This is the number of bytes required to move from the start of a pixel 
  //(offset by output_channel_slice_offset) to the adjecent pixel to the 
  //right (offset by output_channel_slice_offset).
  const size_t output_h_mem_stride; //(same for all output region shapes)

  //This is the number of bytes required to move from the start of a pixel 
  //(offset by output_channel_slice_offset) on the final column of a output 
  //region to the first column of the output region pixel on the next line 
  //down(offset by output_channel_slice_offset).
  const size_t output_w_mem_stride; //different for all output regions of different widths

  //Pointer to scratch memory
  int8_t * scratch_mem;

  //All args structs have protected internal orders, i.e. changing the order of 
  //member types will break the asm

  //these params are shared between all parallel instances
  MemCpyFn * memcpy_handler;
  
  //unique to each output slice
  AggregateFn * aggregate_handler;

  OutputTransformFn * ot_handler;

  public:
    Filter2D();
    void execute (int8_t * Y, int8_t * X);
    
  private:
    void calc_output_pixel_slice(int8_t * Y, int8_t * X, int32_t h, int32_t w);

};