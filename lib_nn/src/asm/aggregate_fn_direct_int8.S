// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_input_channel_loop_counter 3
#define PARAMS_inner_x_h_step 4
#define PARAMS_inner_x_v_step 5
#define PARAMS_weights_bytes 6
#define PARAMS_weights 7


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


/*
extern "C" void mat_mul_impl_asm(void *params, VPURingBuffer *A, int8_t *X, int32_t output_channel_group);
*/

#define FUNCTION_NAME mat_mul_int8_direct_impl_asm

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ;\
                              { ldw ddr_spill, x[0]; nop}
#define NSTACKWORDS  8
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#define NSTACKWORDS  6
#endif
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    #ifdef USE_DDR_FIX
    {stw  ddr_spill, sp[6]; nop}
    #endif

    {vclrdr; ldc t, 32}
    {ldw kw, params_p[PARAMS_bytes_per_kernel_channel]; shl t, t, 4}
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
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
                OPTIONAL_DDR_SPILL(K_p)
                {vlmaccr0 K_p[0]; add K_p, K_p, thirty_two}
            {bt ic, ic_loop_asm; sub ic, ic, 1}
        {bt kw, kw_loop_asm; sub kw, kw, 1}
    {bt kh, kh_loop_asm; sub kh, kh, 1}

    {vstr A_p[0]; add A_p, A_p, thirty_two}
    {vstd A_p[0]; nop}

    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    #ifdef USE_DDR_FIX
    {ldw  ddr_spill, sp[6]; nop}
    #endif
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

#define FUNCTION_NAME mat_mul_int8_direct_impl_asm

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { x.ldwi ddr_spill, 8(x); x.nop} ;\
                              { x.ldwi ddr_spill, 0(x); x.nop}
#define NSTACKWORDS  16
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#define NSTACKWORDS  16
#endif
    
.text
.globl FUNCTION_NAME
.p2align 2
.type FUNCTION_NAME,@function

FUNCTION_NAME:
    x.entsp (NSTACKWORDS)*4

    x.stdsp  x19,x18,0
    x.stdsp  x21,x20,8
    x.stdsp  x23,x22,16
    #ifdef USE_DDR_FIX
    {x.stwsp  ddr_spill, 24; x.nop}
    #endif

    {x.vclrdr; x.ldcu t, 32}
    {x.ldwi kw, (PARAMS_bytes_per_kernel_channel)*4(params_p); x.shli t, t, 4}

//error TranslationError, "During operand canonicalisation got:\nAssertion failed:\n  Condition: the simple text value of the item at position 0 in the srcinstruction's operands = 'r11'\n  Failed for: vsetc t\nFor packet: {vsetc t; nop}"
    {x.vsetc t; x.nop}
    mul output_channel_group, kw, output_channel_group
    //{x.ldwsp K_p, (NSTACKWORDS + 1)*4; x.nop}
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
                {x.vlmaccr0 K_p; add K_p, K_p, thirty_two}
            {x.bt ic, ic_loop_asm; subi ic, ic, 1}
        {x.bt kw, kw_loop_asm; subi kw, kw, 1}
    {x.bt kh, kh_loop_asm; subi kh, kh, 1}

    {x.vstr A_p; add A_p, A_p, thirty_two}
    {x.vstd A_p; x.nop}

    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.lddsp  x23,x22,16
    #ifdef USE_DDR_FIX
    {x.ldwsp  ddr_spill, 24; x.nop}
    #endif
    x.retsp (NSTACKWORDS)*4

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif





