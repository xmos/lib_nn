
// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__XS3A__)

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

#ifndef XCORE
  #define FUNCTION_NAME add_elementwise
#else
  #define FUNCTION_NAME add_elementwise_asm
#endif // NN_USE_REF

#define NSTACKWORDS     (7 + 16 + 1) // 7 used + 16 for temp vecs + 1 to make even
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define PARAMS_M1         0
#define PARAMS_M2         8
#define PARAMS_SHR        16
#define PARAMS_BIAS_HI    24
#define PARAMS_BIAS_LO    32

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
    dualentsp NSTACKWORDS
    // save stack
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    stw r10, sp[6]

    // load vectors from params struct
    ldc tmp1, PARAMS_BIAS_HI
    ldaw bias_hi, arg_add_params_p[tmp1]
    ldc tmp1, PARAMS_BIAS_LO
    ldaw bias_lo, arg_add_params_p[tmp1]
    ldc tmp1, PARAMS_M1
    ldaw m1, arg_add_params_p[tmp1]
    ldc tmp1, PARAMS_M2
    ldaw m2, arg_add_params_p[tmp1]
    ldc tmp1, PARAMS_SHR
    ldaw shift, arg_add_params_p[tmp1]

    // set vector mode to 8 bit
    ldc r11, 32
    shl r11, r11, 4
    {vsetc r11}

    // set up loop counter
    ldw tmp2, sp[STACK_OUT_COUNT]
    shr counter, tmp2, 4

    // init start position
    ldw tmp1, sp[STACK_OUT_START]
    add arg_x1_p, arg_x1_p, tmp1
    add arg_x2_p, arg_x2_p, tmp1
    add arg_y_p, arg_y_p, tmp1

  label_loop_main:
    // x1 and x2 are 8 bit inputs
    // we load them and macc with vpu_vects_vec_0x01
    // to extend them into 16 bit
    vclrdr
    ldap r11, vpu_vects_vec_0x01
    ldaw tmp1, sp[STACK_VEC1_TMP]
    ldaw tmp2, sp[STACK_VEC2_TMP]
    vldc r11[0]
    vlmacc arg_x1_p[0]
    vstr tmp1[0]
    vclrdr
    vlmacc arg_x2_p[0]
    vstr tmp2[0]

    // load bias into acc
    // vldr - vR can only be loaded from r11
    mov r11, bias_lo
    vldr r11[0]
    vldd bias_hi[0]

    // set vector mode to 16 bit
    ldc r11, 32
    shl r11, r11, 3
    {vsetc r11}

    vldc tmp1[0]
    vlmacc m1[0]

    vldc tmp2[0]
    vlmacc m2[0]

    // set vector mode from 16 to 8 bit before vlsat
    shl r11, r11, 1
    {vsetc r11}

    vlsat shift[0]

    // set default mask to 16 elements
    mkmsk tmp1, 16
    bt counter, label_store_with_mask
    // in last loop - make mask for remaining elements
    ldw tmp2, sp[STACK_OUT_COUNT]
    zext tmp2, 5
    mkmsk tmp1, tmp2

 label_store_with_mask:
    vstrpv arg_y_p[0], tmp1

    // increment pointers by 16 elements
    ldc tmp1, 16
    add arg_y_p, arg_y_p, tmp1
    add arg_x1_p, arg_x1_p, tmp1
    add arg_x2_p, arg_x2_p, tmp1

    {sub counter, counter, 1; bt counter, label_loop_main}

.Lfunc_end:
    // restore stack
    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    ldw r10, sp[6]
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

