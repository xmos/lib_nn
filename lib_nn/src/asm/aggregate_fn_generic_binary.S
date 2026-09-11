// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_output_slice_channel_count 0
#define PARAMS_bytes_per_kernel_channel 1

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
extern "C" void mat_mul_generic_binary_impl_asm(MatMulInt8::Params *params, VPURingBuffer *A, int8_t *X, int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_generic_binary_impl_asm


#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ;\
                              { ldw ddr_spill, x[0]; nop}
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif

#define NSTACKWORDS  6
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16                           // This aligns input_channel_loop
.skip   0
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define VPU_INT16_EPV 16

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
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, thirty_two}

        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, step}

    {bt ic_group_count, input_channel_loop_asm; sub ic_group_count, ic_group_count, 1}
    
    calc_step_asm:
    {ashr step, step, 5}
    {ldc t, (VPU_INT16_EPV-1)}
    {add step, step, t; nop}
    {sub step, step, 1; nop} //subtract one for now 

    {vldc X_p[0]; nop}
.align 4
    tail_loop_asm:
        OPTIONAL_DDR_SPILL(K_p)
        {vlmaccr1 K_p[0]; add K_p, K_p, K_p_adjust}
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

#define FUNCTION_NAME mat_mul_generic_binary_impl_asm
#define NSTACKWORDS 8

#define PARAMS_output_slice_channel_count 0
#define PARAMS_bytes_per_kernel_channel 1

#define params_p          a0
#define A_p               a1
#define X_p               a2
#define out_ch_gr         a3
#define K_p               out_ch_gr
#define step              s2
#define K_p_adjust        s3
#define ic_gr_count       s4
#define thirty_two        a0

#define VPU_INT16_EPV 16

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    { li t3, 32 ; xm.entsp (NSTACKWORDS * 4) }
    xm.stdsp s2, s3, 0
    xm.ldd step, K_p_adjust, PARAMS_output_slice_channel_count(params_p)
    { li thirty_two, 15 ; xm.stwsp s4, 8 }

    { xm.shl t3, t3, 4 ; xm.vclrdr }
    { xm.shl out_ch_gr, out_ch_gr, 4 ; xm.vsetc t3 }

    mul t3, K_p_adjust, out_ch_gr
    xm.sub step, step, out_ch_gr
    add K_p, a4, 0
    { add K_p, K_p, t3 ; sub t3, step, thirty_two }

    { xm.shr ic_gr_count, K_p_adjust, 5 ; xm.zext K_p_adjust, 5 }
    { xm.bt K_p_adjust, .K_p_adjust; li thirty_two, 1 }
    { li K_p_adjust, 32 ; addi ic_gr_count, ic_gr_count, -1 }

.K_p_adjust:
    li step, 32
    bge t3, thirty_two, .step_adjust
    mul step, step, t3

.step_adjust:
    li thirty_two, 32
    { xm.brff ic_gr_count, .calc_step ; addi ic_gr_count, ic_gr_count, -1 }

.input_channel_loop:
    { xm.vldr X_p ; add X_p, X_p, thirty_two }

    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }
    { xm.vlmaccrb K_p ; add K_p, K_p, thirty_two }

    { xm.vlmaccrb K_p ; add K_p, K_p, step }

    { xm.bt ic_gr_count, .input_channel_loop ; addi ic_gr_count, ic_gr_count, -1}

.calc_step:
    li t3, -5
    { li t3, (VPU_INT16_EPV - 1 - 1) ; xm.shl step, step, t3 } // subtracting one more
    { add step, step, t3 ; xm.vldc X_p }

.tail_loop:
    { xm.vlmaccrb K_p ; add K_p, K_p, K_p_adjust }
    { xm.bt step, .tail_loop ; addi step, step, -1 }

    { xm.vstr A_p ; add A_p, A_p, thirty_two }
    xm.vstd A_p

    xm.lddsp s2, s3, 0
    xm.ldwsp s4, 8
    xm.retsp (NSTACKWORDS * 4)
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
