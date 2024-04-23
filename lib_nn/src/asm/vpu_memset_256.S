// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_USE_REF
#include <xs1.h>

// void vpu_memset_256(void * dst, const void * src, unsigned byte_count);

#define FUNCTION_NAME vpu_memset_256
    
#define NSTACKWORDS     0

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME

    .globl FUNCTION_NAME
    .align 16
    .issue_mode dual

FUNCTION_NAME: 
    { dualentsp NSTACKWORDS        ; add r11, r1, 0 }
    { vldr   r11[0]                ; add r1, r0, 0 }
    { zext  r1, 2                  ; shr r3, r2, 5 }  // r3: vectors
    { bf  r1, .Laligned            ; mkmsk r11, r2 }

    
    { shl   r11, r11, r1           ; sub   r2, r2, 4 }
    { sub   r0, r0, r1             ; add   r2, r2, r1}

    vstrpv r0[0], r11
    add   r0, r0, 4

    ashr  r3, r2, 32
    { bt    r3, .Ldone             ; shr   r3, r2, 5 }

.Laligned:
    { bf  r3, .Ldone_whole_vectors ; shl r11, r3, 5 }
    { ldc r11, 32                  ; sub r2, r2, r11}

.Lloop:
    { vstr r0[0]                   ; sub r3, r3, 1}
    { bt r3, .Lloop                ; add r0, r0, r11}

.Ldone_whole_vectors:
    { mkmsk r2, r2                  ; nop }
    vstrpv r0[0], r2
.Ldone:
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
