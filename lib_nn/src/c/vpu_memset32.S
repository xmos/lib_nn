// Copyright 2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <xs1.h>

// r0 - dest addr            must be word aligned
// r1 - src value            word to be stored
// r2 - number of BLOBS:     0 or more
.globl vpu_memset32_asm
.globl vpu_memset32_asm.nstackwords
.linkset vpu_memset32_asm.nstackwords,8
.align 16
.issue_mode dual

vpu_memset32_asm:
    dualentsp 8
    std r1, r1, sp[0]
    std r1, r1, sp[1]
    { ldaw r11, sp[0]      ; bf r2, early_exit }
    std r1, r1, sp[2]
    std r1, r1, sp[3]
    vldr r11[0]          
    { ldc  r3, 32          ; sub r2, r2, 1     } // number of bytes address increments between vec accesses 


OMC_set_loop:
    { vstr r0[0]            ; add  r0, r0, r3 }
    { bt   r2, OMC_set_loop ; sub r2, r2, 1  }

early_exit: 
	retsp 8

