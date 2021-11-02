// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_inner_x_h_step 3
#define PARAMS_inner_x_v_step 4
#define PARAMS_weights_bytes 5
#define PARAMS_weights 6

//Registers

#define params_p             r0
#define A_p                  r1

#define X_p                  r2
#define output_channel_group r3 
#define kh output_channel_group 

#define kw                   r4
#define ddr_spill            r5

#define sixteen              r6
#define inner_x_h_step       r7

#define inner_x_v_step       r8
#define kw_init              r9

#define t                    r11
#define K_p                  t


/*
void mat_mul_direct_dw_impl_asm(MatMulDirectFn_DW::Params *params,
                                      VPURingBuffer *A, int8_t *X,
                                      int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_direct_dw_impl_asm

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
.align 2
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]

    {vclrdr; ldc t, 32}
    {ldw kw, params_p[PARAMS_bytes_per_kernel_channel]; shl t, t, 4}
    {vsetc t; nop}
    mul output_channel_group, kw, output_channel_group
    {ldw K_p, sp[NSTACKWORDS+1]; nop}
    {ldw inner_x_v_step, params_p[PARAMS_inner_x_v_step]; add K_p, output_channel_group, K_p}
    {ldc sixteen, 16; ldw inner_x_h_step, params_p[PARAMS_inner_x_h_step]}
    {ldw kh, params_p[PARAMS_k_height_loop_counter];nop}
    {ldw kw_init, params_p[PARAMS_k_width_loop_counter]; sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {mov kw, kw_init; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {vldc X_p[0]; add X_p, X_p, inner_x_h_step}
            OPTIONAL_DDR_SPILL(K_p)
            {vlmacc K_p[0]; add K_p, K_p, sixteen}
        {bt kw, kw_loop_asm; sub kw, kw, 1}
    {bt kh, kh_loop_asm; sub kh, kh, 1}

    {shl sixteen, sixteen, 1; nop}
    {vstr A_p[0]; add A_p, A_p, sixteen}
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



