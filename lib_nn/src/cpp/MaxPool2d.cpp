
#include "MaxPool2d.hpp"

#include <iostream>
#include <algorithm>
#include <cstring>

using namespace nn;

constexpr int MaxPool2d_Generic::ChannelsPerOutputGroup;
constexpr int MaxPool2d_Valid::ChannelsPerOutputGroup;


MaxPool2d_Generic::MaxPool2d_Generic(AbstractKernel::Params* ak_params,
                                     ImToColPadded* memcopy_handler,
                                     MaxPoolPatchFn* aggregate_handler,
                                     DirectWriteOutputTransform* ot_handler,
                                     int8_t* scratch_mem)
    : Filter2D_DW( ak_params, memcopy_handler, aggregate_handler, ot_handler, scratch_mem)
{

}

MaxPool2d_Valid::MaxPool2d_Valid(AbstractKernel::Params* ak_params,
                                 DerefInputFn* memcopy_handler,
                                 MaxPoolDirectValidFn* aggregate_handler,
                                 DirectWriteOutputTransform* ot_handler)
    : Filter2D_DW( ak_params, memcopy_handler, aggregate_handler, ot_handler, (int8_t*)nullptr)
{

}