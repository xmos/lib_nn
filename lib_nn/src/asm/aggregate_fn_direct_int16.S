// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

/*
void mat_mul_direct_int16_impl_asm(
    const mat_mul_direct_params_t *params,
    VPURingBuffer *A, 
    int16_t *X,
    int32_t output_channel_group,
    int16_t *kp
);
*/

.issue_mode  dual

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_input_channel_loop_counter 3
#define PARAMS_inner_x_h_step 4
#define PARAMS_inner_x_v_step 5

//Registers
#define params_p             r0
#define A_p                  r1
#define X_p                  r2
#define output_channel_group r3 
#define kh output_channel_group 

#define kw                   r4
#define ic                   r5
#define thirty_two           r6
#define inner_x_h_step       r7
#define inner_x_v_step       r8

#define kw_init              r9
#define ic_init              params_p
#define ddr_spill            r10
#define t                    r11
#define K_p                  t

#define FUNCTION_NAME mat_mul_direct_int16_impl_asm
#define NSTACKWORDS  6
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.skip   0                               // This aligns the ic_loop
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
    {ldw K_p, sp[NSTACKWORDS + 1]; nop}
    {ldw inner_x_v_step, params_p[PARAMS_inner_x_v_step]; add K_p, output_channel_group, K_p}
    {ldc thirty_two, 32; ldw inner_x_h_step, params_p[PARAMS_inner_x_h_step]}
    {ldw kh, params_p[PARAMS_k_height_loop_counter];nop}
    {ldw kw_init, params_p[PARAMS_k_width_loop_counter]; sub X_p, X_p, inner_x_h_step}
    {ldw ic_init, params_p[PARAMS_input_channel_loop_counter]; sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {mov kw, kw_init; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {mov ic, ic_init; add X_p, X_p, inner_x_h_step}
            ic_loop_asm:
                {vldc X_p[0]; add X_p, X_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
                {vlmaccr K_p[0]; add K_p, K_p, thirty_two}
            {bt ic, ic_loop_asm; sub ic, ic, 1}
        {bt kw, kw_loop_asm; sub kw, kw, 1}
    {bt kh, kh_loop_asm; sub kh, kh, 1}

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

/*
void mat_mul_direct_int16_impl_asm(
    const mat_mul_direct_params_t *params,
    VPURingBuffer *A, 
    int16_t *X,
    int32_t output_channel_group,
    int16_t *kp
);
*/

#define FUNCTION_NAME   mat_mul_direct_int16_impl_asm
#define NSTACKWORDS     (8)
#define NSTACKBYTES     (NSTACKWORDS * 4)

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_input_channel_loop_counter 3
#define PARAMS_inner_x_h_step 4
#define PARAMS_inner_x_v_step 5

#define params_p             a0
#define A_p                  a1
#define X_p                  a2
#define out_ch_g             a3 
#define kp_init              a4

#define kw                   s2
#define ic                   s3
#define thirty_two           s4
#define inner_x_h_step       s5
#define inner_x_v_step       s6
#define kw_init              s7
#define kp2                  s8

#define tmp                  x28
#define kp                   tmp

#define kh                   out_ch_g 
#define ic_init              params_p

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    xm.entsp NSTACKBYTES
    xm.stdsp s2, s3, 0*8
    xm.stdsp s4, s5, 1*8
    xm.stdsp s6, s7, 2*8
    sw s8, 3*8(sp)
    
    {xm.vclrdr; xm.ldcu tmp, 32}
    {xm.ldwi kw, PARAMS_bytes_per_kernel_channel*4(params_p); xm.shli tmp, tmp, 3}
    xm.vsetc tmp;

    mul out_ch_g, kw, out_ch_g
    mv kp, kp_init 
    {xm.ldwi inner_x_v_step, PARAMS_inner_x_v_step*4(params_p);         xm.add kp, out_ch_g, kp}
    {xm.ldwi inner_x_h_step, PARAMS_inner_x_h_step*4(params_p);         xm.ldcu thirty_two, 32}
    {xm.ldwi kh, PARAMS_k_height_loop_counter*4(params_p);              xm.sub X_p, X_p, inner_x_h_step}
    {xm.ldwi kw_init, PARAMS_k_width_loop_counter*4(params_p);          xm.sub X_p, X_p, inner_x_v_step}
    {xm.ldwi ic_init, PARAMS_input_channel_loop_counter*4(params_p);    xm.add kh, kh, 1}

    
    kh_loop_asm:
        {xm.addi kw, kw_init, 1; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {xm.addi ic, ic_init, 1; add X_p, X_p, inner_x_h_step}
            ic_loop_asm:
                {xm.vldc X_p; xm.add X_p, X_p, thirty_two}                

                .rept 8
                {xm.vlmaccr0 kp; xm.add kp2, kp, thirty_two}
                xm.vlmaccr1 kp
                {xm.vlmaccr0 kp2; xm.add kp, kp2, thirty_two}
                xm.vlmaccr1 kp2
                .endr
                
                xm.subi ic, ic, 1
                bnez ic, ic_loop_asm

            xm.subi kw, kw, 1
            bnez kw, kw_loop_asm

    xm.subi kh, kh, 1
    bnez kh, kh_loop_asm

    {xm.vstr A_p; xm.add A_p, A_p, thirty_two}
    xm.vstd A_p
    xm.lddsp s2, s3, 0*8
    xm.lddsp s4, s5, 1*8
    xm.lddsp s6, s7, 2*8
    lw s8, 3*8(sp)
    xm.retsp NSTACKBYTES

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", NSTACKBYTES
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
