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
  //(offset by output_channel_slice_offset) on the final column of a output 
  //region to the first column of the output region pixel on the next line 
  //down(offset by output_channel_slice_offset).
  const size_t output_w_mem_stride; //different for all output regions of different widths

};

template <class T>
class AbstractKernel {
  protected:
  
    AbstractKernelParams kparams;

  public:
    //calc_output_pixel_slice(TOutput *Y, TInput *X, int32_t h, int32_t w);
    void execute (int8_t * Y, int8_t * X) ;
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
      AggregateFn * aggregate_handler, OutputTransformFn * ot_handler, int8_t * scratch_mem);

    void inline calc_output_pixel_slice(int8_t *Y, int8_t *X, int32_t h, int32_t w) ;
};

  
// struct Conv2DParams{

//   //needs to implement []

//   int x_height;
//   int x_width;
//   int x_channels;

//   int k_height;
//   int k_width;
//   int k_channels;

//   int k_dilation_h;
//   int k_dilation_v;

//   int k_stride_h;
//   int k_stride_v;

//   public:
//     int& operator[](int idx){
//       return 0;
//     };
// };

// template <class Tparams>
// class ParamSequence {
//   private:
//     Tparams state_;
//     const Tparams step_;
//     const Tparams min_;
//     const Tparams max_;

//   public:

//     ParamSequence(Tparams min, Tparams max, Tparams step) {
//       state_ = min;
//     };

//     void next(){
//       state_[0] += step_[0];
//       int index = 0;
//       while (state_[index] >= max[index]){
//         state_[index] = min[index];
//         index++;
//         state_[index] += step_[index];
//       }
//     }

//     Tparams* operator++() {
//       state_ = min + step_;
//     }; 

// }
