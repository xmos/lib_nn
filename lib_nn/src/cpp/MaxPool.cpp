// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <cstdint>
#include <cstring>

#include "AggregateFn.hpp"

using namespace nn;

/*
Performs a maxpool operation on the input tensor X, storing the result in the accumulator A.
The operation is defined by the parameters in 'params', which specify the kernel size and others. 

Note: this could live in a low-level C API, but assembly would need changes
to take MaxPool parameters instead of nn::mat_mul_dw_direct_params_t and
VPURingBuffer.
*/

#ifdef NN_USE_REF
void maxpool_direct_ref(
    const mat_mul_dw_direct_params_t *params,
    VPURingBuffer *A, 
    int8_t *X)
{
    int8_t *x_ptr = X;
    int8_t *max = ((int8_t *)&A->vR);
    std::memset(max, -128, VPU_INT16_EPV);
    for (int kh = params->k_height_loop_counter; kh >= 0; kh--)
    {
        for (int kw = params->k_width_loop_counter; kw >= 0; kw--)
        {
            for (int i = 0; i < 16; i++)
            {
                int8_t val = x_ptr[i];
                if (max[i] < val)
                {
                    max[i] = val;
                }
            }
            x_ptr += params->inner_x_h_step;
        }
        x_ptr += params->inner_x_v_step;
    }
}
#else
extern "C" void maxpool_direct_asm(
    const mat_mul_dw_direct_params_t *params,
    VPURingBuffer *A,
    int8_t *X);
#endif

void nn::maxpool_direct(const mat_mul_dw_direct_params_t *params,
                        VPURingBuffer *A,
                        int8_t *T)
{
#ifdef NN_USE_REF
    maxpool_direct_ref(params, A, T);
#else
    maxpool_direct_asm(params, A, T);
#endif
}
