
#include "AvgPool2d.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace nn;

constexpr int AvgPool2d_Generic::ChannelsPerOutputGroup;
constexpr int AvgPool2d_Valid::ChannelsPerOutputGroup;


AvgPool2d_Generic::AvgPool2d_Generic(AbstractKernel::Params* ak_params,
                                     ImToColPadded* memcopy_handler,
                                     AvgPoolPatchFn* aggregate_handler,
                                     ShiftInt8OutputTransform* ot_handler,
                                     int8_t* scratch_mem)
    : Filter2D_DW( ak_params, memcopy_handler, aggregate_handler, ot_handler, scratch_mem)
{

}

AvgPool2d_Valid::AvgPool2d_Valid(AbstractKernel::Params* ak_params,
                                 DerefInputFn* memcopy_handler,
                                 AvgPoolDirectValidFn* aggregate_handler,
                                 ShiftInt8OutputTransform* ot_handler)
    : Filter2D_DW( ak_params, memcopy_handler, aggregate_handler, ot_handler, (int8_t*)nullptr)
{

}