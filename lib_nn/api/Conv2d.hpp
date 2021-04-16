#include <tuple>
#include <cmath>

#include "Filter2D.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"

namespace nn {
namespace filt2d {

class Conv2dVaildDirect : public Filter2D {


  public:

  // void make(
  //   const nn::Filter2dGeometry& filter_geometry,
  //   const ImageRegion &ir,
  //   const int8_t input_img[],
  //   const int8_t kernel_weights[],
  //   const int32_t biases[],
  //   const float effective_output_multiplier[],
  //   const int8_t input_zero_point,
  //   const int8_t output_zero_point,

  //    AbstractKernel::Params &akp, 
  //    ImToColValid::Params &memcpy,
  //     MatMulDirectFn::Params & aggregator, 
  //     OT_int8 &ot ,
  //     std::vector<int16_t> boggled_biases, 

  //   );

    // Conv2dVaildDirect( AbstractKernel::Params * akp, ImToColValid::Params * memcpy,
    //   MatMulDirectFn::Params * aggregator, OT_int8 * ot): Filter2D(akp, memcpy, aggregator, ot){};



    // Filter2D(AbstractKernel::Params * kparams, 
    //          MemCpyFn * memcpy_handler, 
    //          AggregateFn * aggregate_handler, 
    //          OutputTransformFn * ot_handler, 
    //          int8_t * scratch_mem=nullptr);

};



// class Conv2dVaildIndirect : public Filter2D {
//   public:
//     Conv2dVaildIndirect(AbstractKernelParams * kparams, ImToColValid * memcpy_handler, 
//       MatMulInt8 * aggregate_handler, OT_int8 * ot_handler);
// };



// class Conv2dPaddedIndirect : public Filter2D {
//   public:
//     Conv2dPaddedIndirect(AbstractKernelParams * kparams, ImToColPadded * memcpy_handler, 
//       MatMulInt8 * aggregate_handler, OT_int8 * ot_handler);
// };
}
}
