#include <tuple>
#include <cmath>

#include "Filter2D.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"

namespace nn {

  class Conv2dVaildDirect : public Filter2D {
    
    //Valid only
    //Multiple of 32 input and 16 output channels
    Conv2dVaildDirect( 
        AbstractKernel::Params * akp, 
        DerefInputFn * memcpy,
        MatMulDirectFn * aggregator, 
        OT_int8 * ot):
      Filter2D(akp, memcpy, aggregator, ot) {}
  };

  class Conv2dVaildIndirect : public Filter2D {

    //Valid only
    //Arbitrary input + output channel count
    Conv2dVaildIndirect( 
        AbstractKernel::Params * akp, 
        ImToColValid * memcpy,
        MatMulInt8 * aggregator, 
        OT_int8 * ot):
      Filter2D(akp, memcpy, aggregator, ot) {}
  };

  class Conv2dPaddedInDirect : public Filter2D {

    //Padded
    //Arbitrary input + output channel count
    Conv2dPaddedInDirect(
        AbstractKernel::Params * akp, 
        ImToColPadded * memcpy,
        MatMulInt8 * aggregator, 
        OT_int8 * ot):
      Filter2D(akp, memcpy, aggregator, ot) {}
  };

}
