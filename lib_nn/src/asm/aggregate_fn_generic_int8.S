// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_output_slice_channel_count 0
#define PARAMS_bytes_per_kernel_channel 1
#define PARAMS_weights 2


//Registers

#define params_p          r0
#define A_p               r1
#define X_p               r2
#define output_channel_group r3 
#define K_p output_channel_group 
#define K_p_adjust        r4 
#define ic_group_count    r5
#define thirty_two        r6
#define step              r7

#define ddr_spill         r8
#define t                 r11

/*
extern "C" void mat_mul_impl_asm(MatMulInt8::Params *params, VPURingBuffer *A, int8_t *X, int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_generic_int8_impl_asm


#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ;\
                              { ldw ddr_spill, x[0]; nop}
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif

#define NSTACKWORDS  40
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16                           // This aligns input_channel_loop
.skip   0
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define VPU_INT16_EPV 16
#define XS3_VPU_VREG_WIDTH_BYTES 32

.align 4

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]

    {vclrdr; ldc t, 32}
    {nop; shl t, t, 4}
    {vsetc t; nop}

    {ldw step, params_p[PARAMS_output_slice_channel_count]; ldc thirty_two, 15}
    {ldw K_p_adjust, params_p[PARAMS_bytes_per_kernel_channel]; shl output_channel_group, output_channel_group, 4}
    
    mul t, K_p_adjust, output_channel_group
    {ldw K_p, sp[NSTACKWORDS+1]; sub step, step, output_channel_group}
    {add K_p, K_p, t; sub t, step, thirty_two}
    {shr ic_group_count, K_p_adjust, 5; zext K_p_adjust, 5}
    {bt K_p_adjust, K_p_adjust_lbl_asm; ldc thirty_two, 1}
        {ldc K_p_adjust, 32; sub ic_group_count, ic_group_count, 1}
    K_p_adjust_lbl_asm:

    {lss thirty_two, t, thirty_two; ldc step, 32}
    bf thirty_two, step_adjust_lbl_asm; 
        mul step, step, t 
    step_adjust_lbl_asm:
    {ldc thirty_two, 32;}
    {bf ic_group_count, calc_step_asm; sub ic_group_count, ic_group_count, 1}
    input_channel_loop_asm:

        {vldc X_p[0]; add X_p, X_p, thirty_two}
        
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, thirty_two}

        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, step}

    {bt ic_group_count, input_channel_loop_asm; sub ic_group_count, ic_group_count, 1}
    
    calc_step_asm:
    {ashr step, step, 5}
    {ldc t, (VPU_INT16_EPV-1)}
    {add step, step, t; nop}
    {sub step, step, 1; nop} //subtract one for now 

    {vldc X_p[0]; nop}
    tail_loop_asm:
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr K_p[0]; add K_p, K_p, K_p_adjust}
    {bt step, tail_loop_asm; sub step, step, 1}

    {vstr A_p[0]; add A_p, A_p, thirty_two}
    {vstd A_p[0]; nop}

    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    retsp NSTACKWORDS

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif





#if defined(__VX4A__) || defined(__VX4B__)


#define FUNCTION_NAME mat_mul_generic_int8_impl_asm

#define NSTACKWORDS  8
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)


#define PARAMS_output_slice_channel_count 0
#define PARAMS_bytes_per_kernel_channel 1
#define PARAMS_weights 2


//Registers

#define params_p          x10
#define A_p               x11
#define X_p               x12
#define output_channel_group x13 
#define K_p               x23 
#define K_p_adjust        x18 
#define ic_group_count    x19
#define thirty_two        x20
#define step              x21

#define ddr_spill         x22
#define t                 x28

#define VPU_INT16_EPV 16


/*
 void mat_mul_generic_int8_impl_asm(
    const void *params,
    void *A, 
    int8_t *X,
    int32_t output_channel_group,
    int8_t *weights)
*/

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}
    xm.stdsp x18, x19, 0*8
    xm.stdsp x20, x21, 1*8
    xm.stdsp x22, x23, 2*8


    {xm.vclrdr; xm.ldcu t, 32}
    {xm.nop; xm.shli t, t, 4}
    {xm.vsetc t; xm.nop}

    {xm.ldwi step, PARAMS_output_slice_channel_count*4(params_p); xm.ldcu thirty_two, 15}
    {xm.ldwi K_p_adjust, PARAMS_bytes_per_kernel_channel*4(params_p); xm.shli output_channel_group, output_channel_group, 4}
    addi K_p, x14, 0
    mul t, K_p_adjust, output_channel_group
    {xm.nop; xm.sub step, step, output_channel_group}
    {xm.add K_p, K_p, t; xm.sub t, step, thirty_two}
    {xm.shri ic_group_count, K_p_adjust, 5; xm.zexti K_p_adjust, 5}
    {xm.bt K_p_adjust, K_p_adjust_lbl_asm; xm.ldcu thirty_two, 1}
        {xm.ldcu K_p_adjust, 32; xm.subi ic_group_count, ic_group_count, 1}
    K_p_adjust_lbl_asm:

    {xm.slt thirty_two, t, thirty_two; xm.ldcu step, 32}
    xm.brff thirty_two, step_adjust_lbl_asm; 
        mul step, step, t 
    step_adjust_lbl_asm:
    {xm.ldcu thirty_two, 32; xm.nop}
    {xm.brff ic_group_count, calc_step_asm; xm.subi ic_group_count, ic_group_count, 1}
    input_channel_loop_asm:

        {xm.vldc X_p; xm.add X_p, X_p, thirty_two}
        
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, thirty_two}

        {xm.vlmaccr0 K_p; xm.add K_p, K_p, step}

    {xm.bt ic_group_count, input_channel_loop_asm; xm.subi ic_group_count, ic_group_count, 1}
    
    calc_step_asm:
    srai step, step, 5
    li t, (VPU_INT16_EPV-1)
    add step, step, t
    {xm.subi step, step, 1; xm.nop} //subtract one for now 

    {xm.vldc X_p; xm.nop}
    tail_loop_asm:
        {xm.vlmaccr0 K_p; xm.add K_p, K_p, K_p_adjust}
    {xm.bt step, tail_loop_asm; xm.subi step, step, 1}

    {xm.vstr A_p; xm.add A_p, A_p, thirty_two}
    {xm.vstd A_p; xm.nop}

    xm.lddsp x18, x19, 0*8
    xm.lddsp x20, x21, 1*8
    xm.lddsp x22, x23, 2*8

    {xm.retsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif


