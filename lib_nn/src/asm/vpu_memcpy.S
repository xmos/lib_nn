// Copyright 2020-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

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
.skip   4               // This aligns loop
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

#endif

















#if defined(__VX4A__)

#define FUNCTION_NAME vpu_memcpy_vector_ext_asm

#define NSTACKWORDS (6)
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

//    void vpu_memcpy_vector_ext_asm(
//        void* dst, 
//        const void* src, 
//        size_t byte_count);

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

    {xm.ldcu x13, 32; xm.brff x12, early_exit } // number of bytes address increments between vec accesses 
    xm.stdsp x18, x19, 0*8
    xm.stdsp x20, x21, 1*8

    xm.ldcu  x18, 64       // number of bytes address increments, i.e. 2 lines, between 2 stages of loop 
    xm.ldcu  x19, 68       // 2 lines + 1word offset to do misaligned prefetch of 2 lines, 2 lines ahead 
    xm.ldcu  x20, 128      // 4 lines, the number of bytes to increment addresses per loop

    add x28, x11, 4        // get the first 2 lines, for 1st iteration
    { xm.prefetch x28       ; xm.add  x28, x11, x19  }

vpu_memcpy_vector_ext_asm_OMC_cpy_loop:
    { xm.prefetch x28       ; xm.add  x28, x28, x18 }   // prefetch 2 lines for second half of loop
    addi  x28, x11, 0
    
    { xm.vldc x28           ; xm.add  x28, x28, x13 }
    { xm.vldd x28           ; xm.addi x28, x10, 0   }
    { xm.vstc x28           ; xm.add  x28, x28, x13 }
    { xm.vstd x28           ; xm.add  x28, x11, x19 }
    
    add x28, x28, x18
    { xm.prefetch x28    ; xm.add  x28, x11, x18 }
    { xm.vldc x28        ; xm.add  x28, x28, x13 }
    { xm.vldd x28        ; xm.add  x28, x10, x18 }
    { xm.vstc x28        ; xm.add  x28, x28, x13 }
    { xm.vstd x28        ; xm.subi x12, x12, 1   }

    { xm.add  x11, x11, x20     ; xm.add x10, x10, x20  } 
    {xm.bt x12, vpu_memcpy_vector_ext_asm_OMC_cpy_loop; xm.add x28, x11, x19  }

    xm.lddsp x18, x19, 0*8
    xm.lddsp x20, x21, 1*8

early_exit: 
    {xm.retsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"


#undef FUNCTION_NAME
#undef NSTACKWORDS

#define FUNCTION_NAME vpu_memcpy_vector_int_asm
#define NSTACKWORDS (8)

//    void vpu_memcpy_vector_int_asm(
//        void* dst, 
//        const void* src, 
//        size_t byte_count);

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.subi x12, x12, 1; xm.brff x12, dvpu_memcpy_vector_int_asm_done}
    {xm.ldcu x28, 32;xm.nop}

vpu_memcpy_vector_int_asm_loop:
    {xm.vldd x11; xm.add x11, x11, x28}
    {xm.vstd x10; xm.add x10, x10, x28}
    {xm.bt x12, vpu_memcpy_vector_int_asm_loop; xm.subi x12, x12, 1}
dvpu_memcpy_vector_int_asm_done:
	ret

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"



#endif
