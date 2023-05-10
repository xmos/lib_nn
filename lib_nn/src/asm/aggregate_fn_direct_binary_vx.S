// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)


#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_input_channel_loop_counter 3
#define PARAMS_inner_x_h_step 4
#define PARAMS_inner_x_v_step 5
#define PARAMS_weights_bytes 6
#define PARAMS_weights 7


//Registers

#define params_p             x10
#define A_p                  x11
#define X_p                  x12
#define output_channel_group x13 
#define kh output_channel_group 

#define kw                   x18
#define ic                   x19
#define thirty_two           x20
#define inner_x_h_step       x21
#define inner_x_v_step       x22

#define kw_init              x23
#define ic_init              params_p
#define ddr_spill            x24
#define t                    x28
#define K_p                  t


/*
extern "C" void mat_mul_impl_asm(void *params, VPURingBuffer *A, int8_t *X, int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_binary_direct_impl_asm
#define NSTACKWORDS  16
    
.text
.globl FUNCTION_NAME
.p2align 2
.type FUNCTION_NAME,@function

FUNCTION_NAME:
    x.entsp (NSTACKWORDS)*4

    x.stdsp  x19,x18,0
    x.stdsp  x21,x20,8
    x.stdsp  x23,x22,16

    {x.vclrdr; x.ldcu t, 32}
    {x.ldwi kw, (PARAMS_bytes_per_kernel_channel)*4(params_p); x.shli t, t, 4}
    {x.vsetc t; x.nop}
    mul output_channel_group, kw, output_channel_group
    addi K_p, x14, 0
    {x.ldwi inner_x_v_step, (PARAMS_inner_x_v_step)*4(params_p); add K_p, output_channel_group, K_p}
    {x.ldcu thirty_two, 32; x.ldwi inner_x_h_step, (PARAMS_inner_x_h_step)*4(params_p)}
    {x.ldwi kh, (PARAMS_k_height_loop_counter)*4(params_p);x.nop}
    {x.ldwi kw_init, (PARAMS_k_width_loop_counter)*4(params_p); sub X_p, X_p, inner_x_h_step}
    {x.ldwi ic_init, (PARAMS_input_channel_loop_counter)*4(params_p); sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {addi kw, kw_init,0; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {addi ic, ic_init,0; add X_p, X_p, inner_x_h_step}
            ic_loop_asm:
                {x.vldc X_p; add X_p, X_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
                {x.vlmaccrb K_p; add K_p, K_p, thirty_two}
            {x.bt ic, ic_loop_asm; subi ic, ic, 1}
        {x.bt kw, kw_loop_asm; subi kw, kw, 1}
    {x.bt kh, kh_loop_asm; subi kh, kh, 1}

    {x.vstr A_p; add A_p, A_p, thirty_two}
    {x.vstd A_p; x.nop}

    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.lddsp  x23,x22,16
    x.retsp (NSTACKWORDS)*4

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif



