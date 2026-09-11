// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>

    
// multiply_int16_tensor_asm(int16_t *output, int16_t *input, int16_t *input2, int elements, char *blob);

#define FUNCTION_NAME multiply_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     8
#define ARGUMENT4       (NSTACKWORDS+1)
    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: input2
    // r3: number of elements,
    // r4: blob, then rounding value
    // r5, r6: temporary values
    // r7, r8: temporary values high/low accumulator
    // r9: shift - 16
    // r10: multiplier
    // r11: shift
    .align 16
    .skip  4
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]
    std   r8, r9, sp[2]
    stw   r10, sp[6]
    { bf    r3, skip              ; nop }
    { ldw   r4, sp[ARGUMENT4]     ; sub r3, r3, 1     }
    { ldw   r10, r4[0]            ; ldc  r5, 16       }
    { ldc   r4, 1                 ; shr  r11, r10, 16  }
    { sub   r9, r11, r5           ; sub    r5, r11, 1 }
    { shl   r4, r4, r5            ; sext  r10, 16 }
multiply_tensor_16_loop:
    { ld16s r5, r2[r3]            ; ldc r7, 0 }
    { ld16s r6, r1[r3]            ; add r8, r4, 0 }
    mul     r5, r5, r6
    maccs   r7, r8, r5, r10
    lsats   r7, r8, r9
    lextract r7, r7, r8, r11, 16
    st16    r7, r0[r3]
    { bt    r3, multiply_tensor_16_loop ; sub r3, r3, 1 }
skip:   

    ldd   r4, r5, sp[0]
    ldd   r6, r7, sp[1]
    ldd   r8, r9, sp[2]
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
.Ltmp1:
    .size FUNCTION_NAME, .Ltmp1-FUNCTION_NAME

#endif

#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME multiply_int16_tensor_asm
#define NSTACKWORDS 8
    // a0: output
    // a1: input1
    // a2: input2
    // a3: number of elements,
    // a4: blob, then temp

    // s2: rounding value
    // s3, s4, s5: temporary values
    // s6: shift - 16
    // s7: multiplier
    // t3: shift
.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    xm.entsp (NSTACKWORDS * 4)
    xm.stdsp s2, s3, 0
    xm.stdsp s4, s5, 8
    xm.stdsp s6, s7, 16
    { xm.brff a3, .skip ; addi a3, a3, -1 }
    lw s7, 0(a4)
    { li s3, 16 ; xm.shr t3, s7, 16 }
    { sub s6, t3, s3 ; addi s3, t3, -1 }
    { li s2, 1 ; xm.sext s7, 16 }
    xm.shl s2, s2, s3

.loop:
    xm.ld16s s3, a3(a2)
    xm.ld16s a4, a3(a1)
    { li s4, 0 ; add s5, s2, 0 }
    mul s3, s3, a4
    xm.maccs s4, s5, s3, s7
    xm.lsats s4, s5, s6
    xm.lextract s4, s4, s5, t3, 16
    xm.st16 s4, a3(a0)
    { addi a3, a3, -1 ; xm.bt a3, .loop }

.skip:
    xm.lddsp s2, s3, 0
    xm.lddsp s4, s5, 8
    xm.lddsp s6, s7, 16
    xm.retsp (NSTACKWORDS * 4)
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
