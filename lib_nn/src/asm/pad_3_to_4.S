    
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

pad_3_to_4_asm:   
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    { stw r10, sp[6]              ; mkmsk r4, 24 }
    { andnot r3, r4               ; nop }
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


#if defined(__VX4A__)
    // This function is the optimal FIR on a 1-bit signal with 16-bit coefficients.
    // x10: argument 1, output 
    // x11: argument 2, input 
    // x12: argument 3, count (3 byte chunks)
    // x13: argument 4, pad_val(first 8 bits)
#define NSTACKWORDS   16
// void pad_3_to_4_run(int8_t outputs[], int8_t inputs[], uint32_t N_3, uint32_t pad_val) {

.globl pad_3_to_4_run
pad_3_to_4_run:

    x.entsp NSTACKWORDS
    x.stdsp x18, x19, 0
    {subi x12, x12, 1           ; x.mkmski x18, 24 }
    slli x13, x13, 24
loop:
    {x.ldwi x19, 0(x11)   ; addi x11, x11, 3 }
    and x19, x19, x18
    or x19, x19, x13
    { x.stwi x19, 0(x10)  ; addi x10, x10, 4 }
    { x.bt x12, loop      ; subi x12, x12, 1 }

    x.lddsp x18, x19, 0
    x.retsp NSTACKWORDS

#endif

