// Copyright 2020-2025 XMOS LIMITED.
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
void mat_mul_dw_direct_impl_asm(
                const void *params,
                void *A, 
                int8_t *X,
                int32_t output_channel_group,
                int8_t *weights){}
*/

#define FUNCTION_NAME mat_mul_dw_direct_impl_asm

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








#if defined(__VX4A__)


#define FUNCTION_NAME mat_mul_dw_direct_impl_asm

#define NSTACKWORDS  8
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_inner_x_h_step 3
#define PARAMS_inner_x_v_step 4
#define PARAMS_weights_bytes 5
#define PARAMS_weights 6

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
void mat_mul_dw_direct_impl_asm(
                const void *params,
                void *A, 
                int8_t *X,
                int32_t output_channel_group,
                int8_t *weights){}
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
    {xm.ldwi kw, PARAMS_bytes_per_kernel_channel*4(params_p); xm.shli t, t, 4}
    {xm.vsetc t; xm.nop}
    mul output_channel_group, kw, output_channel_group
    addi K_p, x14, 0
    {xm.ldwi inner_x_v_step, PARAMS_inner_x_v_step*4(params_p); xm.add K_p, output_channel_group, K_p}
    {xm.ldcu sixteen, 16; xm.ldwi inner_x_h_step, PARAMS_inner_x_h_step*4(params_p)}
    {xm.ldwi kh, PARAMS_k_height_loop_counter*4(params_p); xm.nop}
    {xm.ldwi kw_init, PARAMS_k_width_loop_counter*4(params_p); xm.sub X_p, X_p, inner_x_v_step}

    kh_loop_asm:
        {xm.addi kw, kw_init, 0; xm.add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            {xm.vldc X_p; xm.add X_p, X_p, inner_x_h_step}
            {xm.vlmacc0 K_p; xm.add K_p, K_p, sixteen}
        {xm.bt kw, kw_loop_asm; xm.subi kw, kw, 1}
    {xm.bt kh, kh_loop_asm; xm.subi kh, kh, 1}

    {xm.shli sixteen, sixteen, 1; xm.nop}
    {xm.vstr A_p; xm.add A_p, A_p, sixteen}
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

#if defined(__VX4A__)

#undef FUNCTION_NAME
#define FUNCTION_NAME mat_mul_dw_direct_row_accum_impl_asm

#define NSTACKWORDS_ROWACC 10
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

// void mat_mul_dw_direct_row_accum_impl_asm(
//     const void *params,
//     void *A,
//     int8_t *X,
//     int32_t output_channel_group,
//     int8_t *weights,
//     int32_t output_width,
//     int32_t x_stride_bytes,
//     int32_t output_stride);

#define params_p             x10
#define A_p                  x11
#define X_p                  x12
#define output_channel_group x13

#define K_base               x14
#define output_width         x15
#define x_stride_bytes       x16
#define output_stride        x17

#define kw                   x18
#define kh                   x19
#define inner_x_h_step       x20
#define inner_x_v_step       x21
#define kw_init              x22
#define kh_init              x23

#define K_p                  x24
#define tmp                  x28

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS_ROWACC*4, 16); xm.nop}
    xm.stdsp x18, x19, 0*8
    xm.stdsp x20, x21, 1*8
    xm.stdsp x22, x23, 2*8

    // Set VPU to 8-bit mode
    {xm.vclrdr; xm.ldcu tmp, 32}
    {xm.shli tmp, tmp, 4; xm.nop}
    {xm.vsetc tmp; xm.nop}

    // Load args from stack
    {xm.ldwi output_width, (NSTACKWORDS_ROWACC+2)*4(sp); xm.nop}
    {xm.ldwi x_stride_bytes, (NSTACKWORDS_ROWACC+3)*4(sp); xm.nop}
    {xm.ldwi output_stride, (NSTACKWORDS_ROWACC+4)*4(sp); xm.nop}
    {xm.shli output_stride, output_stride, 6; xm.nop}

    // Load params
    {xm.ldwi inner_x_h_step, PARAMS_inner_x_h_step*4(params_p); xm.nop}
    {xm.ldwi inner_x_v_step, PARAMS_inner_x_v_step*4(params_p); xm.nop}
    {xm.ldwi kh_init, PARAMS_k_height_loop_counter*4(params_p); xm.nop}
    {xm.ldwi kw_init, PARAMS_k_width_loop_counter*4(params_p); xm.nop}

    // Compute K_base = weights + bytes_per_kernel_channel_group * output_channel_group
    {xm.ldwi tmp, PARAMS_bytes_per_kernel_channel*4(params_p); xm.nop}
    mul tmp, tmp, output_channel_group
    {xm.ldwi K_base, (NSTACKWORDS_ROWACC+1)*4(sp); xm.nop}
    xm.add K_base, K_base, tmp

    // output_width counter is count-1 for bt loop
    xm.subi output_width, output_width, 1

output_loop_vx4a:
    // Load existing accumulator
    {xm.addi tmp, A_p, 0; xm.ldcu K_p, 32}
    {xm.vldr tmp; xm.add tmp, tmp, K_p}
    {xm.vldd tmp; xm.nop}

    // Set K_p and X_cur_p
    {xm.addi K_p, K_base, 0; xm.addi tmp, X_p, 0}

    // kh loop
    xm.addi kh, kh_init, 0
kh_loop_vx4a:
    xm.addi kw, kw_init, 0
kw_loop_vx4a:
        {xm.vldc tmp; xm.add tmp, tmp, inner_x_h_step}
        {xm.vlmacc0 K_p; xm.add K_p, K_p, 16}
    {xm.bt kw, kw_loop_vx4a; xm.subi kw, kw, 1}
    {xm.add tmp, tmp, inner_x_v_step; xm.nop}
    {xm.bt kh, kh_loop_vx4a; xm.subi kh, kh, 1}

    // Store accumulator
    {xm.addi tmp, A_p, 0; xm.ldcu K_p, 32}
    {xm.vstr tmp; xm.add tmp, tmp, K_p}
    {xm.vstd tmp; xm.nop}

    // Advance pointers for next output
    {xm.add X_p, X_p, x_stride_bytes; xm.nop}
    {xm.add A_p, A_p, output_stride; xm.nop}

    {xm.bt output_width, output_loop_vx4a; xm.subi output_width, output_width, 1}

    xm.lddsp x18, x19, 0*8
    xm.lddsp x20, x21, 1*8
    xm.lddsp x22, x23, 2*8

    {xm.retsp ROUND_UP(NSTACKWORDS_ROWACC*4, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKWORDS_ROWACC*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif

#if defined(__XS3A__)

#undef FUNCTION_NAME
#define FUNCTION_NAME mat_mul_dw_direct_row_accum_impl_asm

#define S_X_STRIDE (7)
#define S_OUT_STRIDE (8)
#define S_KH_INIT (9)
#define NSTACKWORDS_ROWACC 12

// void mat_mul_dw_direct_row_accum_impl_asm(
//     const void *params,
//     void *A,
//     int8_t *X,
//     int32_t output_channel_group,
//     int8_t *weights,
//     int32_t output_width,
//     int32_t x_stride_bytes,
//     int32_t output_stride);

.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.skip  12
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS_ROWACC

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    stw r10, sp[6]

    // Set VPU to 8-bit mode
    {vclrdr; ldc r11, 32}
    {shl r11, r11, 4; nop}
    {vsetc r11; nop}

    // Load args from stack
    {ldw r4, sp[NSTACKWORDS_ROWACC+2]; nop}  // output_width
    {ldw r5, sp[NSTACKWORDS_ROWACC+3]; nop}  // x_stride_bytes
    {ldw r11, sp[NSTACKWORDS_ROWACC+4]; nop} // output_stride
    {shl r11, r11, 6; nop}                  // output_stride_bytes
    {stw r5, sp[S_X_STRIDE]; nop}
    {stw r11, sp[S_OUT_STRIDE]; nop}

    // Load params
    {ldw r7, params_p[PARAMS_inner_x_h_step]; nop}
    {ldw r8, params_p[PARAMS_inner_x_v_step]; nop}
    {ldw r9, params_p[PARAMS_k_height_loop_counter]; nop}
    {ldw r10, params_p[PARAMS_k_width_loop_counter]; nop}
    {stw r9, sp[S_KH_INIT]; ldc r6, 16}

    // Compute K_base = weights + bytes_per_kernel_channel_group * output_channel_group
    {ldw r11, params_p[PARAMS_bytes_per_kernel_channel]; nop}
    mul r11, r11, output_channel_group
    {ldw r3, sp[NSTACKWORDS_ROWACC+1]; nop} // weights
    add r3, r3, r11

    // output_width counter (r4) is count-1 for bt loop
    sub r4, r4, 1

output_loop_rowacc:
    // Load existing accumulator
    {mov r11, A_p; ldc r0, 32}
    {vldr r11[0]; add r11, r11, r0}
    {vldd r11[0]; nop}

    // Set K_p and X_cur_p
    {mov r0, r3; mov r11, X_p}

    // kh loop
    ldw r9, sp[S_KH_INIT]
kh_loop_rowacc:
    mov r5, r10
kw_loop_rowacc:
        {vldc r11[0]; add r11, r11, r7}
        OPTIONAL_DDR_SPILL(r0)
        {vlmacc r0[0]; add r0, r0, r6}
    {bt r5, kw_loop_rowacc; sub r5, r5, 1}
    {add r11, r11, r8; nop}
    {bt r9, kh_loop_rowacc; sub r9, r9, 1}

    // Store accumulator
    {mov r11, A_p; ldc r0, 32}
    {vstr r11[0]; add r11, r11, r0}
    {vstd r11[0]; nop}

    // Advance pointers for next output
    {ldw r5, sp[S_X_STRIDE]; nop}
    {add X_p, X_p, r5; nop}
    {ldw r5, sp[S_OUT_STRIDE]; nop}
    {add A_p, A_p, r5; nop}

    {bt r4, output_loop_rowacc; sub r4, r4, 1}

    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    ldw r10, sp[6]

    retsp NSTACKWORDS_ROWACC

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS_ROWACC
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0_rowacc:
    .size FUNCTION_NAME, .Ltmp0_rowacc-FUNCTION_NAME
    .issue_mode  single

#endif
