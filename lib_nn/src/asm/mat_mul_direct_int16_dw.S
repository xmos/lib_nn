// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_inner_x_h_step 3
#define PARAMS_inner_x_v_step 4

//Registers

#define params_p             r0
#define A_p                  r1

#define X_p                  r2
#define output_channel_group r3 
#define kh output_channel_group 

#define kw                   r4
#define ddr_spill            r5

#define thirtytwo            r6
#define inner_x_h_step       r7

#define inner_x_v_step       r8
#define kw_init              r9

#define t                    r11
#define K_p                  t


/*
void mat_mul_dw_direct_int16_impl_asm(MatMulDirectFn_DW::Params *params,
                                      VPURingBuffer *A, int16_t *X,
                                      int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_dw_direct_int16_impl_asm

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
.align 16                   // This aligns the kh_loop and fully
.skip  12                   // contains the inner loop
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]

    {vclrdr; ldc t, 32}
    {ldw kw, params_p[PARAMS_bytes_per_kernel_channel]; shl t, t, 3}
    {vsetc t; nop}
    mul output_channel_group, kw, output_channel_group
    {ldw K_p, sp[NSTACKWORDS+1]; nop}
    {ldw inner_x_v_step, params_p[PARAMS_inner_x_v_step]; add K_p, output_channel_group, K_p}
    {ldc thirtytwo, 32; ldw inner_x_h_step, params_p[PARAMS_inner_x_h_step]}
    {ldw kh, params_p[PARAMS_k_height_loop_counter];nop}
    {ldw kw_init, params_p[PARAMS_k_width_loop_counter]; sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {mov kw, kw_init; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {vldc X_p[0]; add X_p, X_p, inner_x_h_step}
            OPTIONAL_DDR_SPILL(K_p)
            {vlmacc K_p[0]; add K_p, K_p, thirtytwo}
        {bt kw, kw_loop_asm; sub kw, kw, 1}
    {bt kh, kh_loop_asm; sub kh, kh, 1}

    {vstr A_p[0]; add A_p, A_p, thirtytwo}
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

#define FUNCTION_NAME mat_mul_dw_direct_int16_impl_asm
#define NSTACKWORDS 8

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_inner_x_h_step 3
#define PARAMS_inner_x_v_step 4

//Registers

#define params_p             a0
#define A_p                  a1

#define X_p                  a2
#define out_ch_g             a3

#define kw                   s2
#define kh                   s3

#define inner_x_h_step       s4
#define inner_x_v_step       s5

#define kp                   t3
#define thirtytwo            out_ch_g
#define kw_init              params_p

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    { li t3, 32 ; xm.entsp (NSTACKWORDS * 4) }
    xm.stdsp s2, s3, 0
    xm.stdsp s4, s5, 8

    { xm.shl t3, t3, 3 ; xm.vclrdr }
    xm.vsetc t3
    xm.ldd kw, kh, PARAMS_bytes_per_kernel_channel * 4 (params_p)
    mul out_ch_g, kw, out_ch_g
    mv kp, a4
    { add kp, kp, out_ch_g ; lw inner_x_h_step, PARAMS_inner_x_h_step * 4 (params_p) }
    { li thirtytwo, 32 ; lw inner_x_v_step, PARAMS_inner_x_v_step * 4 (params_p) }
    { sub X_p, X_p, inner_x_v_step ; lw kw_init, PARAMS_k_width_loop_counter * 4 (params_p) }

    kh_loop_asm:
        { add kw, kw_init, 1 ; add X_p, X_p, inner_x_v_step }
        kw_loop_asm:
            { xm.vldc X_p ; add X_p, X_p, inner_x_h_step }

            { xm.vlmacc0 kp ; add kw, kw, -1 }
            xm.vlmacc1 kp

        { add kp, kp, thirtytwo ; xm.bt kw, kw_loop_asm }
    { add kh, kh, -1 ; xm.bt kh, kh_loop_asm }

    { xm.vstr A_p ; add A_p, A_p, thirtytwo }
    xm.vstd A_p

    xm.lddsp s2, s3, 0
    xm.lddsp s4, s5, 8
    xm.retsp (NSTACKWORDS * 4)
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
