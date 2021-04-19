
#include "MaxPool2d.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace nn;

constexpr int MaxPool2d_Generic::ChannelsPerOutputGroup;
constexpr int MaxPool2d_Valid::ChannelsPerOutputGroup;


MaxPool2d_Generic::MaxPool2d_Generic( Params* params,
                                      ImToColPadded* memcopy_handler,
                                      MaxPoolPatchFn* aggregate_handler,
                                      DirectWriteOutputTransform* ot_handler,
                                      int8_t* scratch_mem)
    : Filter2D_DW( &params->ak_params, memcopy_handler, aggregate_handler, ot_handler, scratch_mem)
{
}


// MaxPool2d_Valid MaxPool2dValid::Make(const Params* params,
//                                      int8_t* scratch_mem)
// {
//   auto 
// }


// MaxPool2d_Valid::MaxPool2d_Valid( const Params* params,
//                                   int8_t* scratch_mem)
//     : Filter2D_DW(&params->ak_params, 
//                   &DerefInputFn(&params->mem_params), 
//                   &MaxPoolDirectValidFn(&params->agg_params),
//                   &DirectWriteOutputTransform(&params->ot_params), scratch_mem),
//       params(params)
// {
// }