// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <xs1.h>

// r0 - dest addr            must be word aligned
// r1 - src addr             must be word aligned
// r2 - number of VPU_MEMCPU_BLOBS:   0 or more
.globl vpu_memcpy_vector_ext_asm
.globl vpu_memcpy_vector_ext_asm.nstackwords
.linkset vpu_memcpy_vector_ext_asm.nstackwords,4
.align 16
.issue_mode dual

vpu_memcpy_vector_ext_asm:
    dualentsp 4
    std  r4, r5, sp[0]
    stw  r6,  sp[2]
    
    { ldc  r3, 32          ; bf r2, early_exit } // number of bytes address increments between vec accesses 
    ldc  r4, 64       // number of bytes address increments, i.e. 2 lines, between 2 stages of loop 
    ldc  r5, 68       // 2 lines + 1word offset to do misaligned prefetch of 2 lines, 2 lines ahead 
    ldc  r6, 128      // 4 lines, the number of bytes to increment addresses per loop

    add r11, r1, 4                               // get the first 2 lines, for 1st iteration
    { prefetch r11          ; add  r11, r1, r5  }

OMC_cpy_loop:
    { prefetch r11          ; add  r11, r11, r4 }   // prefetch 2 lines for second half of loop
    mov  r11, r1
    
    { vldc r11[0]           ; add  r11, r11, r3 }
    { vldd r11[0]           ; mov  r11, r0      }
    { vstc r11[0]           ; add  r11, r11, r3 }
    { vstd r11[0]           ; add  r11, r1,  r5 }
    
    add r11, r11, r4
    { prefetch r11          ; add  r11, r1,  r4 }
    { vldc r11[0]           ; add  r11, r11, r3 }
    { vldd r11[0]           ; add  r11, r0,  r4 }
    { vstc r11[0]           ; add  r11, r11, r3 }
    { vstd r11[0]           ; sub  r2,  r2,  1  }

    { add  r1, r1, r6       ; add  r0,  r0, r6  } 
    { bt   r2, OMC_cpy_loop ; add  r11, r1, r5  }

early_exit: 
    ldd r4, r5, sp[0]
    ldw r6, sp[2]
	retsp 4

// void vpu_memcpy_vector_int(void * dst, const void * src, int vector_count);

.globl vpu_memcpy_vector_int_asm
.globl vpu_memcpy_vector_int_asm.nstackwords
.linkset vpu_memcpy_vector_int_asm.nstackwords,4

.align 16

.issue_mode dual

vpu_memcpy_vector_int_asm:
    dualentsp 0
    
    {sub r2, r2, 1; bf r2, done}
    {ldc r11, 32;nop}

loop:
    {vldd r1[0]; add r1, r1, r11}
    {vstd r0[0]; add r0, r0, r11}
    {bt r2, loop; sub r2, r2, 1}
done:
	retsp 0

