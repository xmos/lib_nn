// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)


#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_inner_x_h_step 3
#define PARAMS_inner_x_v_step 4
#define PARAMS_weights_bytes 5
#define PARAMS_weights 6

//Registers

#define params_p             x10
#define A_p                  x11

#define X_p                  x12
#define output_channel_group x13 
#define kh output_channel_group 

#define kw                   x18
#define ddr_spill            x19

#define sixteen              x20
#define inner_x_h_step       x21

#define inner_x_v_step       x22
#define kw_init              x23

#define t                    x28
#define K_p                  t


/*
void mat_mul_direct_dw_impl_asm(MatMulDirectFn_DW::Params *params,
                                      VPURingBuffer *A, int8_t *X,
                                      int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_direct_dw_impl_asm

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { x.ldwu ddr_spill, 8(x); nop} ;\
                              { x.ldwu ddr_spill, 0(x); nop}
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif
    
#define NSTACKWORDS  16
.text
.globl FUNCTION_NAME
.p2align 1
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
    
    //{x.ldwsp K_p, (NSTACKWORDS+1)*4; x.nop}
    addi K_p, x14, 0 
    
    {x.ldwi inner_x_v_step, (PARAMS_inner_x_v_step)*4(params_p); add K_p, output_channel_group, K_p}
    {x.ldcu sixteen, 16; x.ldwi inner_x_h_step, (PARAMS_inner_x_h_step)*4(params_p)}
    {x.ldwi kh, (PARAMS_k_height_loop_counter)*4(params_p);x.nop}
    {x.ldwi kw_init, (PARAMS_k_width_loop_counter)*4(params_p); sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {addi kw, kw_init,0; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {x.vldc X_p; add X_p, X_p, inner_x_h_step}
            OPTIONAL_DDR_SPILL(k_p)
            {x.vlmacc0 K_p; add K_p, K_p, sixteen}
        {x.bt kw, kw_loop_asm; subi kw, kw, 1}
    {x.bt kh, kh_loop_asm; subi kh, kh, 1}

    {x.shli sixteen, sixteen, 1; x.nop}
    {x.vstr A_p; add A_p, A_p, sixteen}
    {x.vstd A_p; x.nop}

    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.lddsp  x23,x22,16


    x.retsp (NSTACKWORDS)*4

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif



