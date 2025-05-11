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