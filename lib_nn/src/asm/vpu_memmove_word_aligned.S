// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <xs1.h>

// void vpu_memmove_word_aligned(void * dst, const void * src, int byte_count);

.globl vpu_memmove_word_aligned
.globl vpu_memmove_word_aligned.nstackwords
.linkset vpu_memmove_word_aligned.nstackwords,0

.align 16
.issue_mode dual

vpu_memmove_word_aligned:
    { dualentsp 0                  ; shr r3, r2, 5}
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

	retsp 0

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

	retsp 0
