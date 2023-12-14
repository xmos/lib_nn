#include <stdio.h>
#include <stdint.h>
#include <math.h>

#include "output_transform_fn_int16.h"
#include "output_transform_fn_int16_kernel_transform.h"
#include "output_transform_fn_int16_mappings.h"

#define VPU_INT16_EPV 16
#define VPU_INT32_EPV 8

int min(int a, int b) {
    return a < b ? a : b;
}
#ifdef NN_USE_REF

int16_t *output_transform_fn_int16_impl(int16_t *vDvR,
                                        int32_t *mul_add,
                                        int16_t *output,
                                        uint32_t N) {
    for(int i = 0; i < N; i++) {
        int32_t multiplier = mul_add[ot_int16_mul_index_used_for_output[i]];
        int32_t adder      = mul_add[ot_int16_add_index_used_for_output[i]];
        int32_t accu_high  = vDvR[i];
        int32_t accu_low   = vDvR[i+16];
        int32_t accu       = (accu_low & 0xffff) | (accu_high << 16);
        int64_t answer     = multiplier * (int64_t) accu;
        answer             = (answer + (1<<29)) >> 30;
        answer            += adder;
        if (answer > 32767) {
            answer = 32767;
        }
        if (answer < -32768) {
            answer = -32768;
        }
        output[i]          = answer;
    }
    return &output[N];
}

#else

extern void output_transform_fn_int16_impl_asm(int16_t *vDvR,
                                               int32_t *mul_add,
                                               int16_t *output,
                                               uint32_t N);
#endif


int16_t *output_transform_fn_int16(otfn_int16_params_t *params,
                                   int16_t *Y,
                                   int16_t *vDvR,
                                   int32_t output_channel_group,
                                   int32_t *mul_add) {
    int output_count = min(
        params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
        (int32_t)VPU_INT16_EPV);
    Y       += output_channel_group * VPU_INT16_EPV;
    mul_add += output_channel_group * VPU_INT32_EPV * 2;
#ifdef NN_USE_REF
    return output_transform_fn_int16_impl(vDvR, mul_add,
                                          Y, output_count);
#else
    output_transform_fn_int16_impl_asm(vDvR, mul_add,
                                       Y, output_count);
    return Y + output_count;
#endif  // NN_USE_REF
}

