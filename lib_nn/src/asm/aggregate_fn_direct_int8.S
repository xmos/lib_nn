// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_weights 0
#define PARAMS_bytes_per_kernel_channel 1
#define PARAMS_k_height_loop_counter 2
#define PARAMS_k_width_loop_counter 3
#define PARAMS_input_channel_loop_counter 4
#define PARAMS_inner_x_h_step 5
#define PARAMS_inner_x_v_step 6

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

#define t                    r11
#define K_p                  t


/*
extern "C" void mat_mul_impl_asm(void *params, vpu_ring_buffer_t *A, int8_t *X, int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_direct_impl_asm

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

     // TODO push the multiply by VPU_INT16_EPV into bytes_per_kernel_channel
    {ldw t, params_p[PARAMS_bytes_per_kernel_channel]; nop }
    mul t, t, output_channel_group
    {ldw K_p, params_p[PARAMS_weights]; shl t, t, 4} 
    {add K_p, t, K_p}

    {ldw inner_x_v_step, params_p[PARAMS_inner_x_v_step];nop} 
    {ldw inner_x_h_step, params_p[PARAMS_inner_x_h_step];nop} 

    ldw kh, params_p[PARAMS_k_height_loop_counter]
    {sub kh, kh, 1; nop} //TODO push the sub into constructor

    ldw kw_init, params_p[PARAMS_k_width_loop_counter]
    ldw ic_init, params_p[PARAMS_input_channel_loop_counter]
    {sub ic_init, ic_init, 1; sub kw_init, kw_init, 1} //TODO push the sub into constructor

    {ldc thirty_two, 32; nop}

    kh_loop:
        {mov kw, kw_init; nop}
        kw_loop:
            {mov ic, ic_init; nop}
            ic_loop:
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

            {bt ic, ic_loop; sub ic, ic, 1}

            {add X_p, X_p, inner_x_h_step; nop}    //this could be hoisted

        {bt kw, kw_loop; sub kw, kw, 1}

        {add X_p, X_p, inner_x_v_step; nop}    //as could this
    {bt kh, kh_loop; sub kh, kh, 1}

    {vstr A_p[0]; add A_p, A_p, thirty_two}
    {vstr A_p[0]; nop}

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



