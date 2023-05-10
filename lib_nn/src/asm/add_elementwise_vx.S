
// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__VX4A__)

/*
void add_asm(
    int8_t *y,
    int8_t *x1,
    int8_t *x2,
    nn_add_params_t *p,
    int output_start,
    int output_end);

typedef struct {
        int16_t m1[16];
        int16_t m2[16];
        int16_t shr[16];
        int16_t bias_hi[16];
        int16_t bias_lo[16];
} nn_add_params_t;
*/

#define FUNCTION_NAME add_elementwise_asm

#define NSTACKWORDS     (7 + 16 + 1) // 7 used + 16 for temp vecs + 1 to make even

.text
.globl FUNCTION_NAME
.p2align 4
.type FUNCTION_NAME,@function

#define PARAMS_M1         0
#define PARAMS_M2         8
#define PARAMS_SHR        16
#define PARAMS_BIAS_HI    24
#define PARAMS_BIAS_LO    32

#define STACK_ORIG_DP     (NSTACKWORDS-17)
#define STACK_VEC1_TMP    (NSTACKWORDS-16)
#define STACK_VEC2_TMP    (NSTACKWORDS-8)
#define STACK_OUT_START   (NSTACKWORDS+1)
#define STACK_OUT_COUNT   (NSTACKWORDS+2)

#define arg_y_p           r0
#define arg_x1_p          r1
#define arg_x2_p          r2
#define arg_add_params_p  r3

#define tmp1              r4
#define tmp2              r5
#define counter           arg_add_params_p

#define bias_lo           r6
#define bias_hi           r7
#define m1                r8
#define m2                r9
#define shift             r10

.skip 8
FUNCTION_NAME:

    x.entsp (NSTACKWORDS)*4
/*
    // save stack
    x.stdsp  x19,x18,0
    x.stdsp  x21,x20,8
    x.stdsp  x23,x22,16
    x.stwsp x24, 24

    // Load vectors from params struct
//error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  stwsp => V: No encoding available\n  ldc => S: Okay\nSwapped order failed because:\n  ldc => V: Okay\n  stwsp => S: Blocked by format override: stwsp_sru6 can only store general purpose registers\nFor packet: {stwsp x26, STACK_ORIG_DP        ;     ldc x28, 16}"
    {stwsp x26, STACK_ORIG_DP        ;     ldc x28, 16}
    // Subtract 16 here from output pointer as an optmization
    // to add 16 before use in loop_main
    sub arg_y_p, arg_y_p, x28
error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  setdp => V: No encoding available\n  shli => S: Okay\nSwapped order failed because:\n  shli => V: Okay\n  setdp => S: No encoding available\nFor packet: {setdp  arg_add_params_p         ;     shli x28, x28, 5}"
error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  ldawdp => V: No encoding available\n  ldawdp => S: No encoding available\nSwapped order failed because:\n  ldawdp => V: No encoding available\n  ldawdp => S: No encoding available\nFor packet: {ldawdp bias_hi, PARAMS_BIAS_HI ;     ldawdp bias_lo, PARAMS_BIAS_LO}"
error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  ldawdp => V: No encoding available\n  ldawdp => S: No encoding available\nSwapped order failed because:\n  ldawdp => V: No encoding available\n  ldawdp => S: No encoding available\nFor packet: {ldawdp m1, PARAMS_M1           ;     ldawdp m2, PARAMS_M2}"
    // Load and init start position
error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  ldawdp => V: No encoding available\n  ldwsp => S: Okay\nSwapped order failed because:\n  ldwsp => V: No encoding available\n  ldawdp => S: No encoding available\nFor packet: {ldawdp shift, PARAMS_SHR       ;     ldwsp tmp1, STACK_OUT_START}"
    // Set vector mode to 8 bit and set up loop counter
    {x.ldwsp tmp2, (STACK_OUT_COUNT)*4    ;     add arg_x1_p, arg_x1_p, tmp1}
    {x.vsetc x28;     x.shri counter, tmp2, 4}
    {add arg_x2_p, arg_x2_p, tmp1     ;     add arg_y_p, arg_y_p, tmp1}

  label_loop_main:
    // x1 and x2 are 8 bit inputs
    // We load them and macc with vpu_vects_vec_0x01
    // to extend them into 16 bit
    x.vclrdr
    lla x28, vpu_vects_vec_0x01
error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  ldawsp => V: Okay\n  ldawsp => S: No encoding available\nSwapped order failed because:\n  ldawsp => V: Okay\n  ldawsp => S: No encoding available\nFor packet: {ldawsp tmp1, STACK_VEC1_TMP    ;     ldawsp tmp2, STACK_VEC2_TMP}"
    {x.vldc x28;     x.ldcu x28, 16}
    {x.vlmacc arg_x1_p;     add arg_x1_p, arg_x1_p, x28}
    {x.vstr tmp1;     add arg_y_p, arg_y_p, x28}
    x.vclrdr
    {x.vlmacc arg_x2_p;     add arg_x2_p, arg_x2_p, x28}
    // Load bias into acc
    // vldr - vR can only be loaded from r11
    {x.vstr tmp2;     addi x28, bias_lo,0}
    {x.vldr x28;     x.ldcu x28, 32}
    // Set vector mode to 16 bit
    {x.vldd bias_hi;     x.shli x28, x28, 3}
    x.vsetc x28
    // Macc inputs with multipliers
    x.vldc tmp1
    x.vlmacc m1
    x.vldc tmp2
    x.vlmacc m2

    // Saturation fix
    // Save original 32-bit outputs
    // We saturate the number to 16-bits.
    // We then use the upper 16-bits to find out which numbers are
    // going to be -128.
    // -128's are written to output with a mask.
    // The lower 16-bits are later saturated using vlsat and
    // written with a mask to fill in remaining spaces.
    x.vstd tmp2
    {x.vstr tmp1;     x.ldcu tmp1, 32}
    {x.vlsat shift;     x.shli tmp1, tmp1, 4}
    x.vldd tmp2
    lla x28, vpu_vects_vec_0x007F
    {x.vladd x28;     addi x28, tmp1,0}
    x.vdepth1
    // Switch to 8-bit mode
    x.vsetc x28
#define saturated_mask    tmp2
#define mask              tmp1
    // Store saturated_mask and set default mask to 16 elements
    {x.vstr saturated_mask;     x.mkmski mask, 16}

    bne counter,x0, label_store_with_mask
    // In last loop - make mask for remaining elements
    x.ldwsp tmp2, (STACK_OUT_COUNT)*4
    x.zexti tmp2, 4
    x.mkmski mask, tmp2

  label_store_with_mask:
    x.ldwsp saturated_mask, (STACK_VEC2_TMP)*4
    lla x28, vpu_vects_vec_0x80
    and saturated_mask, saturated_mask, mask
    {x.vldr x28;     x.ldawsp x28, (STACK_VEC1_TMP)*4}
    // Store -128's
    x.vstrpv arg_y_p, saturated_mask

error TranslationError, "During translation got:\nCouldn't find format for packet even after swapping order;\nGiven order failed because:\n  vldr => V: No encoding available\n  andnot => S: Okay\nSwapped order failed because:\n  andnot => V: No encoding available\n  vldr => S: Okay\nFor packet: {vldr ;     andnot mask, saturated_mask}"
    x.vlsat shift
    // Store other elements
    x.vstrpv arg_y_p, mask

    {subi counter, counter, 1          ;     x.bt counter, label_loop_main}

.Lfunc_end:
    // Restore stack
    lw x26, (STACK_ORIG_DP)*4(x2)
    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.lddsp  x23,x22,16
    x.ldwsp x24, 24

    */
    x.retsp (NSTACKWORDS)*4
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif

