// Copyright 2025-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)
   // This function is the optimal FIR on a 1-bit signal with 16-bit coefficients.
    // r0: argument 1, output (word aligned)
    // r1: argument 2, input (word aligned)
    // r2: argument 3, count (number of bytes to be done)
    // r3: argument 4, pad_val(top 24 bits)

#define NSTACKWORDS   4
    
    .globl pad_1_to_4_asm
    .globl pad_1_to_4_asm.nstackwords
    .globl pad_1_to_4_asm.maxthreads
    .globl pad_1_to_4_asm.maxtimers
    .globl pad_1_to_4_asm.maxchanends
    .linkset pad_1_to_4_asm.nstackwords, NSTACKWORDS
    .linkset pad_1_to_4_asm.threads, 0
    .linkset pad_1_to_4_asm.maxtimers, 0
    .linkset pad_1_to_4_asm.chanends, 0

    .cc_top pad_1_to_4_asm.func, pad_1_to_4_asm
    .type pad_1_to_4_asm, @function
    
    .text
    .issue_mode dual
    .align 16

pad_1_to_4_asm:   
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    stw r6, sp[2]
    { ldc r6, 16                  ; mkmsk r5, 8     }
    { andnot r3, r5               ; bu entry        }
loop:
    { stw r4, r0[3]               ; add r0, r0, r6  }
entry:
    { ldw r4, r1[0]               ; add r1, r1, 4   }
    { and r11, r4, r5             ; bf r2, done     }
    { or r11, r11, r3             ; shr r4, r4, 8   }
    { stw r11, r0[0]              ; and r11, r4, r5 }
    { or r11, r11, r3             ; shr r4, r4, 8   }
    { stw r11, r0[1]              ; and r11, r4, r5 }
    { or r11, r11, r3             ; shr r4, r4, 8   }
    { stw r11, r0[2]              ; sub r2, r2, 1   }
    { or r4, r4, r3               ; bu loop         }
done:
    ldd r4, r5, sp[0]
    ldw r6, sp[2]
    retsp NSTACKWORDS
    
    .cc_bottom pad_1_to_4_asm.func
#endif

#if defined(__VX4B__)

.pad_1_to_4_lut1:
    .word 0x030201FF, 0x030201FF, 0x030201FF, 0x030201FF
    .word 0x030201FF, 0x030201FF, 0x030201FF, 0x030201FF

.pad_1_to_4_lut2:
    .word 0xFFFFFF00, 0xFFFFFF01, 0xFFFFFF02, 0xFFFFFF03
    .word 0xFFFFFF04, 0xFFFFFF05, 0xFFFFFF06, 0xFFFFFF07

#define FUNCTION_NAME pad_1_to_4_asm
#define NSTACKWORDS 12

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    { xm.ldap t3, .pad_1_to_4_lut1 ; xm.entsp (NSTACKWORDS * 4) }
    { xm.ldap t3, .pad_1_to_4_lut2 ; xm.vldr t3 }
    { add a3, sp, 0 ; xm.stwsp a3, 0 }
    { xm.shl a2, a2, 2 ; xm.vldc a3 }
    xm.vlookup a1, 0, 0
    { xm.mkmsk t3, 3 ; xm.vldr t3 }
    { and t3, a2, t3 ; xm.shr a2, a2, 3 }
    { addi a1, a1, 8 ; xm.brff a2, .tail }

.loop:
    xm.vlookup a1, 0, 0
    { addi a2, a2, -1 ; xm.vstd a0 }
    add a0, a0, 32 // could load 32 into a3, but had nothing to packet with
    { add a1, a1, 8 ; xm.bt a2, .loop }

.tail:
    { add a3, sp, 0 ; xm.brff t3, .end }

    xm.vlookup a1, 0, 0
    { xm.shl t3, t3, 2 ; xm.vstd a3 }
    { xm.mkmsk t3, t3 ; xm.vldr a3 }
    xm.vstrpv a0, t3

.end:
    xm.retsp (NSTACKWORDS * 4)

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif // defined(__VX4B__)
