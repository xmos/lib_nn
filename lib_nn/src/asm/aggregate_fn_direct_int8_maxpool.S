// Copyright 2020-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/*
void maxpool_direct_impl_asm(MatMulDirectFn_DW::Params *params,
                            VPURingBuffer *A, int8_t *X);
*/

#define FUNCTION_NAME maxpool_direct_impl_asm

#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_unused                0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter  2
#define PARAMS_inner_x_h_step        3
#define PARAMS_inner_x_v_step        4

//Registers


#define t                    r11


#define S_TEMP            6
#define NSTACKWORDS       8

.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]

    {ldw r8, r0[PARAMS_inner_x_v_step]       ; ldc r11, 32     }
    {ldw r7, r0[PARAMS_inner_x_h_step]       ; shl r11, r11, 4 }
    {ldw r3, r0[PARAMS_k_height_loop_counter]; ldc r5, 3       }
    {ldw r9, r0[PARAMS_k_width_loop_counter] ; sub r2, r2, r8}
    {vsetc r11                               ; add r11, r2, r8 }

    { vldr r11[0]                            ; mkmsk  r0, 16 }
    vstrpv r1[0], r0
    { stw r0, sp[S_TEMP]                     ; add r11, r1, 0}       // Clear two bytes
    kh_loop_asm:
        {add r4, r9, 1            ; add r2, r2, r8}
        kw_loop_asm:
            { vldr r11[0]         ; sub r4, r4, 1       }
            { vlsub r2[0]         ; nop                 }
            { vdepth1             ; ldaw r6, sp[S_TEMP] }
              vstrpv r6[0], r5
            { ldw r6, sp[S_TEMP]  ; add r11, r2, 0     }
            xor r6, r6, r0
            { vldr r11[0]         ; add r2, r2, r7     }
              vstrpv r1[0], r6
        {bt r4, kw_loop_asm       ; add r11, r1, 0     }
    {bt r3, kh_loop_asm           ; sub r3, r3, 1}

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

#define NSTACKWORDS  8
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)


#define PARAMS_bytes_per_kernel_channel 0
#define PARAMS_k_height_loop_counter 1
#define PARAMS_k_width_loop_counter 2
#define PARAMS_inner_x_h_step 3
#define PARAMS_inner_x_v_step 4

#define S_TEMP            (4*6)
#define NSTACKWORDS       8

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}
    xm.stdsp x18, x19, 0*8
    xm.stdsp x20, x21, 1*8
    xm.stdsp x22, x23, 2*8

    {xm.ldwi x22, 4*PARAMS_inner_x_v_step(x10)       ; xm.ldcu x28, 32     }
    {xm.ldwi x21, 4*PARAMS_inner_x_h_step(x10)       ; xm.shli x28, x28, 4 }
    {xm.ldwi x13, 4*PARAMS_k_height_loop_counter(x10); xm.ldcu x19, 3       }
    {xm.ldwi x23, 4*PARAMS_k_width_loop_counter(x10) ; sub x12, x12, x22}
    {xm.vsetc x28                                    ; add x28, x12, x22 }

    { xm.vldr x28                                    ; xm.mkmski  x10, 16 }
    xm.vstrpv x11, x10
    { xm.stwsp x10, S_TEMP                           ; xm.addi x28, x11, 0}       // Clear two bytes

    kh_loop_asm:
        {xm.addi x18, x23, 1                         ; xm.add x12, x12, x22     }
        kw_loop_asm:
            { xm.vldr x28                            ; xm.sub x18, x18, 1       }
            { xm.vlsub x12                           ; nop                      }
            { xm.vdepth1                             ; xm.ldawsp x20, S_TEMP   }
            xm.vstrpv x20, x19                                        // store just one word
            { xm.ldwsp x20, S_TEMP                   ; xm.addi x28, x12, 0      }
            xor x20, x20, x10
            { xm.vldr x28                            ; xm.add x12, x12, x21     }
            xm.vstrpv x11, x20
        {xm.bt x18, kw_loop_asm                      ; xm.addi x28, x11, 0      }
    {xm.bt x13, kh_loop_asm                          ; xm.subi x13, x13, 1}

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
