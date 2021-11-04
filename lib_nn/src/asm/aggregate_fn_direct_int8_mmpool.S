// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_k_height_loop_counter 0
#define PARAMS_k_width_loop_counter  1
#define PARAMS_inner_x_h_step        2
#define PARAMS_inner_x_v_step        3
#define PARAMS_minmaxmask            4  // 0 for max, ~0 for min

//Registers

#define params_p             r0
#define A_p                  r1

#define X_p                  r2
#define kh                   r3

#define kw                   r4
//                           r5

//                           r6
#define inner_x_h_step       r7

#define inner_x_v_step       r8
#define kw_init              r9

#define t                    r11


/*
void mmpool_direct_impl_asm(MatMulDirectFn_DW::Params *params,
                            VPURingBuffer *A, int8_t *X);
*/

#define FUNCTION_NAME mmpool_direct_impl_asm

#define S_TEMP            6
#define NSTACKWORDS       8

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

    { vldd X_p[0]               ; nop }
    { vstd A_p[0]               ; nop }

    {ldw inner_x_v_step, r0[PARAMS_inner_x_v_step]       ; ldc r11, 32     }
    {ldw inner_x_h_step, r0[PARAMS_inner_x_h_step]       ; shl r11, r11, 4 }
    {ldw kh,             r0[PARAMS_k_height_loop_counter]; ldc r5, 3       }
    {ldw kw_init,        r0[PARAMS_k_width_loop_counter] ; sub X_p, X_p, inner_x_v_step}
    {ldw r0,             r0[PARAMS_minmaxmask]           ; nop            }
    {vsetc r11                                           ; add r11, A_p, 0}

    kh_loop_asm:
        {add kw, kw_init, 1     ; add X_p, X_p, inner_x_v_step}
        kw_loop_asm:
            { vldr r11[0]       ; sub kw, kw, 1       }
            { vlsub X_p[0]      ; nop                 }
            { vdepth1           ; ldaw r6, sp[S_TEMP] }
              vstrpv r6[0], r5
            { ldw r6, sp[S_TEMP]; add r11, X_p, 0     }
              xor r6, r6, r0
            { vldr r11[0]       ; add X_p, X_p, inner_x_h_step}
              vstrpv A_p[0], r6
        {bt kw, kw_loop_asm     ; add r11, A_p, 0     }
    {bt kh, kh_loop_asm; sub kh, kh, 1}

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



