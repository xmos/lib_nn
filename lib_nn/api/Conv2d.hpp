
#include "Filter2D.hpp"

#include <cmath>


// class Conv2dVaildDirect : public Filter2D {

//   struct Params {
//     AbstractKernel::Params * kparams;
//     ImToColValid::Params * memcpy;
//     MatMulDirectFn::Params * aggregator;
//     // OT_int8::Params * output_transform;
//   };


//   public:
//     Conv2dVaildDirect(Params * params);
// };



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
