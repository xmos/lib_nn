// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>

// requantize_int16_tensor_asm(int16_t *output, int16_t *input, int elements, char *blob);

#define FUNCTION_NAME requantize_int16_tensor_asm

	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    .text
    .issue_mode dual

#define NSTACKWORDS     12
#define ARGUMENT4       (NSTACKWORDS+1)

    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: number of elements, then mask of number of final elements
    // r3: blob, 16 multipliers
    // r5: number of elements / 16
    // r6: temp vector
    // r7: 32
    // r11: temp.
    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 32    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]

    { shr   r5, r2, 4            ; zext  r2, 4     }       // Number of iterations/modulo
    { ldaw  r6, sp[4]            ; shl   r2, r2, 1 }
    { mkmsk r2, r2               ; shl   r11, r11, 3 }     // Make 0x100
    { ldc   r7, 32               ; vsetc r11       }

    ldaw r11, cp[eight_thousand]
    bu    requantise_tensor_16_loop_entry 
requantise_tensor_16_loop:
    { vstd     r0[0]              ; add r0, r0, r7 }
requantise_tensor_16_loop_entry:
    { vldd     r1[0]              ; nop }
    { vldr     r11[0]             ; nop }
    { vldc     r1[0]              ; add r1, r1, r7 }
    { vlmacc   r3[0]              ; nop            }
    { vlmacc   r3[0]              ; nop            }
    
    { bt       r5, requantise_tensor_16_loop ; sub   r5, r5, 1 }
    { vstd     r6[0]              ; add r11, r6, 0 }
    { vldr     r11[0]             ; nop }
    vstrpv   r0[0], r2

    ldd   r4, r5, sp[0]
    ldd   r6, r7, sp[1]

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

#define FUNCTION_NAME requantize_int16_tensor_asm
.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    xm.entsp 0
    xm.ldcu x10, 0
    xm.assert x10 //NOT IMPLEMENTED
    xm.retsp 0
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  0
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
