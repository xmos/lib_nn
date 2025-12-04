// Copyright 2020-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

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




#if defined(__VX4A__) || defined(__VX4B__)
// void vpu_memset_256(void * dst, const void * src, unsigned byte_count);

#define FUNCTION_NAME vpu_memset_256

#define NSTACKWORDS 4
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.addi x28, x11, 0}

    { xm.vldr   x28                        ; xm.addi x11, x10, 0 }
    { xm.zexti  x11, 2                     ; xm.shri x13, x12, 5 }  // x13: vectors
    { xm.brff  x11, vpu_memset_256_aligned ; xm.mkmsk x28, x12 }

    
    { xm.shl   x28, x28, x11               ; xm.subi   x12, x12, 4 }
    { xm.sub   x10, x10, x11               ; xm.add   x12, x12, x11}

    xm.vstrpv x10, x28
    addi   x10, x10, 4

    srai  x13, x12, 31
    { xm.bt    x13, vpu_memset_256_done    ; xm.shri   x13, x12, 5 }

vpu_memset_256_aligned:
    { xm.brff  x13, vpu_memset_256_done_whole_vectors ; xm.shli x28, x13, 5 }
    { xm.ldcu x28, 32                      ; xm.sub x12, x12, x28}

vpu_memset_256_loop:
    { xm.vstr x10                          ; xm.subi x13, x13, 1}
    { xm.bt x13, vpu_memset_256_loop       ; xm.add x10, x10, x28}

vpu_memset_256_done_whole_vectors:
    { xm.mkmsk x12, x12                    ; xm.nop }
    xm.vstrpv x10, x12
vpu_memset_256_done:

	{xm.retsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"


#endif