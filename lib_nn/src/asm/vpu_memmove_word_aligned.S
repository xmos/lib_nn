// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_USE_REF
#include <xs1.h>

// void vpu_memmove_word_aligned(void * dst, const void * src, unsigned byte_count);

#define FUNCTION_NAME vpu_memmove_word_aligned
    
#define NSTACKWORDS     0

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME

    .globl FUNCTION_NAME
    .align 16
    .issue_mode dual

FUNCTION_NAME: 
    { dualentsp NSTACKWORDS        ; shr r3, r2, 5}
    { bf  r3, .Ldone_whole_vectors ; lss r11, r0, r1 }
    { bf r11, .Lcopy_down          ; shl r11, r3, 5 }
    { ldc r11, 32                  ; sub r2, r2, r11}

.Lloop:
    { vldd r1[0]                   ; add r1, r1, r11}
    { vstd r0[0]                   ; sub r3, r3, 1}
    { bt r3, .Lloop                ; add r0, r0, r11}

.Ldone_whole_vectors:
    { add  r11, r1, 0              ; mkmsk r2, r2 }
    { vldr r11[0]                  ; nop }
    vstrpv r0[0], r2

	retsp NSTACKWORDS

.Lcopy_down:
    { sub r2, r2, r11              ; add r0, r0, r11 }
    add r1, r1, r11

    { add  r11, r1, 0              ; mkmsk r2, r2 }
    { vldr r11[0]                  ; ldc r11, 32 }
    vstrpv r0[0], r2
    sub r1, r1, r11
                      

.Lloop_down:
    { vldd r1[0]                   ; sub r0, r0, r11}
    { vstd r0[0]                   ; sub r3, r3, 1}
    { bt r3, .Lloop_down           ; sub r1, r1, r11}

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
