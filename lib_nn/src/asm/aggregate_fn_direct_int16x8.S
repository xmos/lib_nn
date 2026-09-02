// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__) || defined(__VX4B__)

/*
void mat_mul_direct_int16x8_impl_asm(
    const mat_mul_direct_params_t *params,
    VPURingBuffer *A,
    int16_t *T,
    int32_t output_channel_group, 
    int16_t *weights)
*/

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_input_channel_loop_counter 3
#define PARAMS_inner_x_h_step 4
#define PARAMS_inner_x_v_step 5
#define PARAMS_weights_bytes 6
#define PARAMS_weights 7

#define FUNCTION_NAME mat_mul_direct_int16x8_impl_asm

#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

#define NSTACKWORDS  10

#define params_p             x10
#define A_p                  x11
#define X_p                  x12
#define output_channel_group x13 
#define kh output_channel_group 

#define kw                   x18
#define ic                   x19
#define thirty_one           x20
#define inner_x_h_step       x21
#define inner_x_v_step       x22

#define kw_init              x23
#define ic_init              params_p
#define thirty_two           x24
#define t                    x28
#define K_p                  t


.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

    xm.stdsp x18, x19, 0*8
    xm.stdsp x20, x21, 1*8
    xm.stdsp x22, x23, 2*8
    xm.stdsp x24, x25, 3*8
    
    {xm.vclrdr; xm.ldcu t, 32}

    {xm.ldwi kw, PARAMS_bytes_per_kernel_channel*4(params_p); xm.shli t, t, 4}
    {xm.vsetc x28; nop}

    mul output_channel_group, kw, output_channel_group
    addi K_p, x14, 0

    {xm.ldwi inner_x_v_step, PARAMS_inner_x_v_step*4(params_p); xm.add K_p, output_channel_group, K_p}
    {xm.ldcu thirty_one, 31; xm.ldwi inner_x_h_step, PARAMS_inner_x_h_step*4(params_p)}
    {xm.ldwi kh, PARAMS_k_height_loop_counter*4(params_p);xm.ldcu thirty_two, 32}
    {xm.ldwi kw_init, PARAMS_k_width_loop_counter*4(params_p); xm.sub X_p, X_p, inner_x_h_step}
    {xm.ldwi ic_init, PARAMS_input_channel_loop_counter*4(params_p); xm.sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {xm.addi kw, kw_init, 0; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {xm.addi ic, ic_init, 0; add X_p, X_p, inner_x_h_step}
            ic_loop_asm:
                {xm.vldc X_p; xm.add X_p, X_p, thirty_two}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
                {xm.vlmaccr0 K_p; xm.addi K_p, K_p, 1}
                {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_one}
            {xm.bt ic, ic_loop_asm; xm.subi ic, ic, 1}
        {xm.bt kw, kw_loop_asm; xm.subi kw, kw, 1}
    {xm.bt kh, kh_loop_asm; xm.subi kh, kh, 1}

    {xm.vstr A_p; xm.add A_p, A_p, thirty_two}
    {xm.vstd A_p; xm.nop}

    xm.lddsp x18, x19, 0*8
    xm.lddsp x20, x21, 1*8
    xm.lddsp x22, x23, 2*8
    xm.lddsp x24, x25, 3*8

    {xm.retsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
