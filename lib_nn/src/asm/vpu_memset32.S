// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>

// r0 - dest addr            must be word aligned
// r1 - src value            word to be stored
// r2 - number of BLOBS:     0 or more
.globl vpu_memset32_asm
.globl vpu_memset32_asm.nstackwords
.linkset vpu_memset32_asm.nstackwords,8
.align 16
.skip   0               // This aligns OMC_set_loop
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

#endif




#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME vpu_memset32_asm

#define NSTACKWORDS (10)
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

//    void vpu_memset32_asm(
//        void* dst, 
//        const void* src, 
//        size_t byte_count);

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}
    { xm.ldawsp x28, 2*4    ; xm.brff x12, vpu_memset32_asm_early_exit }
   
    xm.stdsp x11, x11, 1*8
    xm.stdsp x11, x11, 2*8
    xm.stdsp x11, x11, 3*8
    xm.stdsp x11, x11, 4*8
    {xm.vldr x28; xm.nop}          
    {xm.ldcu x13, 32         ; xm.subi x12, x12, 1     } // number of bytes address increments between vec accesses 


vpu_memset32_asm_OMC_set_loop:
    { xm.vstr x10            ; xm.add  x10, x10, x13 }
    { xm.bt  x12, vpu_memset32_asm_OMC_set_loop ; xm.subi x12, x12, 1  }

vpu_memset32_asm_early_exit: 
	{xm.retsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"


#endif
