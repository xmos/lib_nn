
#include "Conv2d.hpp"
#include <cmath>

// Conv2dVaildDirect::foo(ImageParams &X, WindowGeometry &K, ImageRegion &ir, int8_t * weights , std::vector<int32_t> bias )
// {

//   ImageParams Y(X, K, 8);

//   int input_ch_per_output; //?

//   ImToColValid cpy(X, K, input_ch_per_output);

//   size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
//   int overread_bytes = cpy.get_overread_bytes(); 

//   ImToColValid * memcpy_handler;

//   MatMulDirectFn mmd(X, K, x_channels, weights);

//   MatMulDirectFn * aggregate_handler;

//   //QuantisationParams qp = OTBinary_int8::quantise_activation(f_multipliers, f_biases, accu_min, accu_max);

//   OT_int8 * ot_handler;

//   AbstractKernel::Params kparams;
//   ImToColValid::Params memcpy_handler_params;
//   MatMulDirectFn::Params aggregate_handler_params;
//   OT_int8::Params ot_handler_params;

//   return ();
// }

// Conv2dVaildDirect::Conv2dVaildDirect(AbstractKernel::Params * kparams, ImToColValid * memcpy_handler, 
//   MatMulDirectFn * aggregate_handler, OT_int8 * ot_handler):Filter2D(kparams, memcpy_handler,
//   aggregate_handler, ot_handler)
// {
  
// }



// Conv2dVaildIndirect::Conv2dVaildIndirect(AbstractKernelParams * kparams, ImToColValid * memcpy_handler, 
//       MatMulInt8 * aggregate_handler, OT_int8 * ot_handler):Filter2D(kparams, memcpy_handler,
//   aggregate_handler, ot_handler)
// {
  
// }

// Conv2dPaddedIndirect::prepare()
// {

// AbstractKernelParams kparams;
// ImToColPadded * memcpy_handler;

// MatMulInt8 * aggregate_handler;
// OT_int8 * ot_handler

// }

// Conv2dPaddedIndirect::Conv2dPaddedIndirect(AbstractKernelParams * kparams, ImToColPadded * memcpy_handler, 
//       MatMulInt8 * aggregate_handler, OT_int8 * ot_handler):Filter2D(kparams, memcpy_handler,
//   aggregate_handler, ot_handler)
// {
  
// }

