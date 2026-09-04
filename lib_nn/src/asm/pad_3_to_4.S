// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

    // This function is the optimal FIR on a 1-bit signal with 16-bit coefficients.
    // r0: argument 1, output (word aligned)
    // r1: argument 2, input (word aligned)
    // r2: argument 3, count (number of 6-word chunks to be done)
    // r3: argument 4, pad_val(first 8 bits)

#define NSTACKWORDS   10
    .globl pad_3_to_4_asm
    .globl pad_3_to_4_asm.nstackwords
    .globl pad_3_to_4_asm.maxthreads
    .globl pad_3_to_4_asm.maxtimers
    .globl pad_3_to_4_asm.maxchanends
    .linkset pad_3_to_4_asm.nstackwords, NSTACKWORDS
    .linkset pad_3_to_4_asm.threads, 0
    .linkset pad_3_to_4_asm.maxtimers, 0
    .linkset pad_3_to_4_asm.chanends, 0

    .cc_top pad_3_to_4_asm.func, pad_3_to_4_asm
    .type pad_3_to_4_asm, @function
    
    .text
    .issue_mode dual
    .align 16
    .skip   4                    // This aligns loop

pad_3_to_4_asm:   
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    { stw r10, sp[6]              ; mkmsk r4, 24 }
    { shl r3, r3, 24              ; nop }
    { ldc r8, 0                   ; ldc r9, 8 }
loop:
    ldd r4, r5, r1[0]
    lextract r7, r4, r5, r8, 24
    { or r7, r7, r3}
    { stw r7, r0[0]               ; ldc r11, 24 }
    lextract r7, r4, r5, r11, 24
    { or r7, r7, r3}
    { stw r7, r0[1]               ; ldc r10, 16 }
    ldd r5, r6, r1[1]
    lextract r7, r6, r4, r10, 24
    { or r7, r7, r3}
    { stw r7, r0[2]               ; nop }
    lextract r7, r5, r6, r9, 24
    { or r7, r7, r3}
    { stw r7, r0[3]               ; nop }
    
    ldd r4, r6, r1[2]
    lextract r7, r6, r5, r8, 24
    { or r7, r7, r3}
    { stw r7, r0[4]               ; add r1, r1, r11 }
    lextract r7, r6, r5, r11, 24
    { or r7, r7, r3}
    { stw r7, r0[5]               ; nop }
    lextract r7, r4, r6, r10, 24
    { or r7, r7, r3}
    { stw r7, r0[6]               ; sub r2, r2, 1 }
    lextract r7, r6, r4, r9, 24
    { or r7, r7, r3}
    { stw r7, r0[7]               ; add r0, r0, r10 }
    { bt r2, loop                 ; add r0, r0, r10 }

    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    ldw r10, sp[6]
    retsp NSTACKWORDS
    
    .cc_bottom pad_3_to_4_asm.func
#endif

#if defined(__VX4B__)

#define FUNCTION_NAME pad_3_to_4_asm
#define NSTACKWORDS 12

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    { xm.shl a3, a3, 24 ; xm.entsp (NSTACKWORDS * 4) }
    { add t3, a3, 0 ; xm.stwsp s2, 32 }
    xm.ldcu s2, 32 // could move before the loop, align it and packet with xm.bu

    xm.stdsp a3, t3, 0
    xm.stdsp a3, t3, 8
    xm.stdsp a3, t3, 16
    xm.stdsp a3, t3, 24
    { add t3, sp, 0 ; xm.mkmsk a3, 3 }

.loop:
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}

    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, t3, 4 ; xm.vstrpv t3, a3}
    { add a1, a1, 3 ; xm.vldr a1 }
    { add t3, sp, 0 ; xm.vstrpv t3, a3}

    { add a2, a2, -1 ; xm.vldr t3 }
    { add a0, a0, s2 ; xm.vstr a0 }
    xm.bt a2, .loop

    xm.ldwsp s2, 32
    xm.retsp (NSTACKWORDS * 4)

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif // defined(__VX4B__)

