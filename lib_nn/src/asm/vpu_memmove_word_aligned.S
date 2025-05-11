// Copyright 2020-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

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



#if defined(__VX4A__)


#define FUNCTION_NAME vpu_memmove_word_aligned

#define NSTACKWORDS  0
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

/*

   void vpu_memmove_word_aligned(
        void * dst,
        const void * src, 
        unsigned byte_count
    );
*/

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.shri x13, x12, 5}
    { xm.brff  x13, vpu_memmove_word_aligned_done_whole_vectors ; xm.slt x28, x10, x11 }
    { xm.brff x28, vpu_memmove_word_aligned_copy_down          ; xm.shli x28, x13, 5 }
    { xm.ldcu x28, 32                  ; xm.sub x12, x12, x28}

vpu_memmove_word_aligned_loop:
    {xm.vldd x11                   ; xm.add x11, x11, x28}
    { xm.vstd x10                   ; xm.subi x13, x13, 1}
    { xm.bt x13, vpu_memmove_word_aligned_loop                ; xm.add x10, x10, x28}

vpu_memmove_word_aligned_done_whole_vectors:
    { xm.addi  x28, x11, 0              ; xm.mkmsk x12, x12 }
    { xm.vldr x28                  ; xm.nop }
    xm.vstrpv x10, x12

	ret

vpu_memmove_word_aligned_copy_down:
    { xm.sub x12, x12, x28              ; xm.add x10, x10, x28 }
    xm.add x11, x11, x28

    { xm.addi  x28, x11, 0              ; xm.mkmsk x12, x12 }
    { xm.vldr x28                  ; xm.ldcu x28, 32 }
    xm.vstrpv x10, x12
    xm.sub x11, x11, x28
                      
vpu_memmove_word_aligned_loop_down:
    { xm.vldd x11                   ; xm.sub x10, x10, x28}
    { xm.vstd x10                  ; xm.subi x13, x13, 1}
    { xm.bt x13, vpu_memmove_word_aligned_loop_down           ; xm.sub x11, x11, x28}

    ret

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif


