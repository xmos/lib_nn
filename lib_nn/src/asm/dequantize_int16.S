// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>
    
// dequantize_int16_tensor_asm(int16_t *output, int16_t *input, int elements, char *blob);

#define FUNCTION_NAME dequantize_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     2

    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output array
    // r1: input array
    // r2: number of elements
    // r3: blob, floating point multiple and floating point addition
    //     0x40008000 afterwards (constant to make int16_t a float)
    // r4: floating point multipler
    // r5: floating point addition
    // r11: temp value
    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop }
    std   r4, r5, sp[0]
    { ldw r4, r3[0]             ; ldap r11, magic_constant }
    { ldw r11, r11[0]           ; nop }
    { ldw r5, r3[1]             ; sub r2, r2, 1 }
dequantize_tensor_16_loop:
    ld16s r3, r1[r2]
    add r3, r3, r11          
    fadd r3, r3, r5
    fmul r3, r3, r4
    stw r3, r0[r2]            
    { bt  r2, dequantize_tensor_16_loop ; sub r2, r2, 1  }

    ldd   r4, r5, sp[0]
    retsp NSTACKWORDS
magic_constant:
    .word 0x40008000
    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp1:
    .size FUNCTION_NAME, .Ltmp1-FUNCTION_NAME

#endif

#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME dequantize_int16_tensor_asm
    // a0: output array
    // a1: input array
    // a2: number of elements
    // a3: blob, floating point multiple and floating point addition
    //     0x40008000 afterwards (constant to make int16_t a float)
    // a4: floating point multipler
    // a5: floating point addition
    // t3: temp value

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    lw a4, 0(a3)
    lw a5, 4(a3)
    li t3, 0x40008000
    addi a2, a2, -1

.loop:
    xm.ld16s a3, a2(a1)
    add a3, a3, t3
    xm.fadd a3, a3, a5
    xm.fmul a3, a3, a4
    xm.stw a3, a2(a0)
    { addi a2, a2, -1 ; xm.bt a2, .loop }

    ret
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  0
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
