
#include "Conv2d.hpp"
#include <cmath>

using namespace nn;
using namespace nn::filt2d;

// std::tuple<ImToColValid::Params, MatMulDirectFn::Params> Conv2dVaildDirect::make(
  
//     const nn::Filter2dGeometry& filter_geometry,
//     const ImageRegion &ir,
//     const int8_t input_img[],

//     const int8_t kernel_weights[],
//     const int32_t biases[],
//     const float effective_output_multiplier[],
//     const int8_t input_zero_point,
//     const int8_t output_zero_point)
// {
//   const ImageGeometry &X = filter_geometry.input;
//   const WindowGeometry &K = filter_geometry.window;
//   const ImageGeometry &Y =  filter_geometry.output;

//   ImToColValid::Params imtocol_params(X, K, X.depth);
//   MatMulDirectFn::Params mmd_params(X, K, X.depth, (int8_t *)kernel_weights);

//   int output_ch_count = 4;
//   int elements_per_channel = 4;
//   OT_int8::Params ot_params(output_ch_count, elements_per_channel, kernel_weights, biases, effective_output_multiplier, input_zero_point, output_zero_point);
//   AbstractKernel::Params akp(Y, ir);


//   // return std::tuple<ImToColValid::Params>(imtocol_params, mmd_params);
// }

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

