// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)


#define PARAMS_output_slice_channel_count 0
#define PARAMS_bytes_per_kernel_channel 1
#define PARAMS_weights 2


//Registers

#define params_p          x10
#define A_p               x11
#define X_p               x12
#define output_channel_group x13 

#define K_p_adjust        x18 
#define ic_group_count    x19
#define thirty_two        x20
#define step              x21
#define K_p               x22

#define ddr_spill         x23
#define t                 x28

/*
C_API void mat_mul_int8_generic_impl_asm(MatMulInt8::Params *params,
                                         VPURingBuffer *A, 
                                         int8_t *X,
                                         int32_t output_channel_group,
                                         int8_t *K_p);
*/

#define FUNCTION_NAME mat_mul_int8_generic_impl_asm


#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { x.ldwi ddr_spill, 8(x); x.nop} ;\
                              { x.ldwi ddr_spill, 0(x); x.nop}
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif

#define NSTACKWORDS  40
    
.text
.globl FUNCTION_NAME
.p2align 1
.type FUNCTION_NAME,@function

#define VPU_INT16_EPV 16
#define XS3_VPU_VREG_WIDTH_BYTES 32

.p2align 2

FUNCTION_NAME:
    x.entsp (NSTACKWORDS)*4

    x.stdsp  x19,x18,0
    x.stdsp  x21,x20,8
    x.stdsp  x23,x22,16

    {x.vclrdr; x.ldcu t, 32}
    sll t, t, 4
    {x.vsetc t; x.nop}
    {x.ldwi step, (PARAMS_output_slice_channel_count)*4(params_p); x.ldcu thirty_two, 15}
    {x.ldwi K_p_adjust, (PARAMS_bytes_per_kernel_channel)*4(params_p); x.shli output_channel_group, output_channel_group, 4}
    
    mul t, K_p_adjust, output_channel_group

    sub step, step, output_channel_group
    mv K_p, x14

    {add K_p, K_p, t; sub t, step, thirty_two}
    {x.shri ic_group_count, K_p_adjust, 5; x.zexti K_p_adjust, 5}
    {x.bt K_p_adjust, K_p_adjust_lbl_asm; x.ldcu thirty_two, 1}
        {x.ldcu K_p_adjust, 32; subi ic_group_count, ic_group_count, 1}
    K_p_adjust_lbl_asm:

    {x.lss thirty_two, t, thirty_two; x.ldcu step, 32}
    {x.brff thirty_two, step_adjust_lbl_asm; x.nop}
        mul step, step, t 
    step_adjust_lbl_asm:
    {x.ldcu thirty_two, 32; x.nop}
    {x.brff ic_group_count, calc_step_asm; subi ic_group_count, ic_group_count, 1}
    input_channel_loop_asm:

        {x.vldc X_p; add X_p, X_p, thirty_two}
        
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}

        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, step}

    {x.bt ic_group_count, input_channel_loop_asm; subi ic_group_count, ic_group_count, 1}
    
    calc_step_asm:
    srai step, step, 5
    {x.ldcu t, (VPU_INT16_EPV-1); x.nop}
    {add step, step, t; x.nop}
    {subi step, step, 1; x.nop} //subtract one for now 

    {x.vldc X_p; x.nop}
    tail_loop_asm:
        OPTIONAL_DDR_SPILL(k_p)
        {x.vlmaccr0 K_p; add K_p, K_p, K_p_adjust}
    {x.bt step, tail_loop_asm; subi step, step, 1}

    {x.vstr A_p; add A_p, A_p, thirty_two}
    {x.vstd A_p; x.nop}

    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.lddsp  x23,x22,16
    x.retsp (NSTACKWORDS)*4

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif



