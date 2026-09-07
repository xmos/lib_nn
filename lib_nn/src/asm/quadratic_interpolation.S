// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

/*
    Code for calculating a quadratic interpolation.
    This function is at present hard coded for 128 chunks of 512 elements each.

    1. Using teh VPU calculate the CHUNK number for each element (0..63)
       and the index withint the CHUNK number (-256..255)

    2. Store this in the output array and on the stack

    3. On the CPU, in a loop, calculate ((A * X + B) * X + C)
       It is assumed that A, B, and C are stored packed as eight bytes
       CC CC CC CC BB BB 00 AA
       C is a signed 32 bit value with the binary point between bits 15 and 16
       B is a signed 32 bit value with the binary point between bits 7 and 8
       A is a signed 8 bit integer.

    4. At the end of the loop, check for a non-multiple of 16 number of remaining elements
       And if so, set up the VSTRPV to write only few elements.
  */ 

    #define NSTACKWORDS             (10+8)
    #define FUNCTION_NAME           quadratic_interpolation_128_asm
    .globl FUNCTION_NAME
    .globl FUNCTION_NAME.nstackwords
    .linkset FUNCTION_NAME.nstackwords, NSTACKWORDS
    .cc_top FUNCTION_NAME.function,FUNCTION_NAME

    .issue_mode dual
    .align 16
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 0x1}
    std   r4, r5, sp[1] 
    std   r6, r7, sp[2]
    { shl r11, r11, 8           ; nop }
    std   r9, r10, sp[3]
    { stw   r8, sp[8]           ; nop }
    { vsetc   r11               ; ldaw r4, sp[10] }
    { shr   r5, r3, 4           ; nop }
    { bf    r5, tail_end        ; nop }              // <= 15 elements, possibly 0
vector_loop:
    { mkmsk r5, 32              ; ldc r6, 15 }       // Default store mask and loop start
vector_loop_last:
    ldc r8, 9
    std   r1, r3, sp[0]
    vlashr r1[0], r8
    { vstr   r4[0]              ; neg r8, r8 }
    vlashr r4[0], r8
    { vlsub  r1[0]              ; ldap r11, zero_offset_9_bit }
    { vladd r11[0]              ; ldap r11, zero_offset_7_bit }
    vstrpv   r0[0], r5
    { vldr r11[0]               ; ldc    r10, 0 }
    { vladd r4[0]               ; ldc    r9, 32 }
    { vstr r4[0]                ; neg    r9, r9 }
    
interpolation_loop:
    { ld16s  r3, r4[r6]         ; ldc r8, 0 }          // Chunk number
    ldd    r3, r5, r2[r3]                              // r3: AB, r5 C, r0 base
    { sext   r3, 8               ; shr    r7, r3, 8 }  // r3 A, r7: B
    { ld16s  r11, r0[r6]         ; shl   r1, r5, r9 }  // r11: X [-256..255], r1 sign of C
    maccs r8, r7, r3, r11                              // {r8,B} += A*X
    maccs r1, r5, r7, r11                              // {r8,C} += B*X
    lsats r1, r5, r10                                  // Saturate C
    { shr   r5, r5, 16            ; nop }              // Shift C down
    st16  r5, r0[r6]                                   // Store output
    { bt    r6, interpolation_loop   ; sub r6, r6, 1 }

    ldc   r5, 16
    ldd   r1, r3, sp[0]
    { sub    r3, r3, r5         ; sub r0, r0, r9 }
    { shr    r5, r3, 4          ; sub r1, r1, r9 }   // Advance in by 32 bytes.
    bt     r5, vector_loop
tail_end:                                            // Almost done <= 15 elements left
    bf     r3, done                                  // Done if zero
    shl    r5, r3, 1                                 // There is a non modulo 16 left.
    mkmsk  r5, r5                                    // This is the vector store mask
    sub    r6, r3, 1                                 // Starting point for last loop
    ldc    r3, 16
    bu     vector_loop_last
done:   
    ldd   r9, r10, sp[3]
    ldd   r4, r5, sp[1] 
    ldd   r6, r7, sp[2] 
    ldw   r8, sp[8] 
    { retsp NSTACKWORDS           ; nop }

zero_offset_7_bit: 
    .word   0x00400040, 0x00400040, 0x00400040, 0x00400040
    .word   0x00400040, 0x00400040, 0x00400040, 0x00400040
zero_offset_9_bit: 
    .word   0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00
    .word   0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00
    
    .cc_bottom FUNCTION_NAME.function
#endif

#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME quadratic_interpolation_128_asm
#define NSTACKWORDS 20

.zero_offset_7_bit:
    .word   0x00400040, 0x00400040, 0x00400040, 0x00400040
    .word   0x00400040, 0x00400040, 0x00400040, 0x00400040
.zero_offset_9_bit:
    .word   0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00
    .word   0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    xm.entsp (NSTACKWORDS * 4)
    xm.stdsp s2, s3, 0
    xm.stdsp s4, s5, 8
    xm.stdsp s6, s7, 16
    li t3, 0x100
    li a4, 0
    { li s7, 32 ; xm.vsetc t3 }
    { xm.shr s3, a3, 4 ; xm.neg s7, s7 }
    { xm.ldawsp s2, (8 * 4) ; xm.brff s3, .tail_end }

.vector_loop:
    { xm.mkmsk s3, 32 ; li s4, 15 }
.vector_loop_last:
    li s6, 9
    xm.stdsp a1, a3, 24
    xm.vlashr a1, s6
    xm.vstr s2
    xm.neg s6, s6
    xm.vlashr s2, s6
    { xm.ldap t3, .zero_offset_9_bit ; xm.vlsub a1 }
    { xm.ldap t3, .zero_offset_7_bit ; xm.vladd t3 }
    xm.vstrpv a0, s3
    xm.vldr t3
    xm.vladd s2
    xm.vstr s2

.interpolation_loop:
    xm.ld16s a3, s4(s2)
    xm.ldd s3, a3, a3(a2)
    { xm.sext a3, 8 ; xm.shr s5, a3, 8 }
    xm.ld16s t3, s4(a0)
    { li s6, 0 ; xm.shl a1, s3, s7 }
    xm.maccs s6, s5, a3, t3
    xm.maccs a1, s3, s5, t3
    xm.lsats a1, s3, a4
    xm.shr s3, s3, 16
    xm.st16 s3, s4(a0)
    { addi s4, s4, -1 ; xm.bt s4, .interpolation_loop }

    li s3, 16
    xm.lddsp a1, a3, 24
    { sub a3, a3, s3 ; sub a0, a0, s7 }
    { xm.shr s3, a3, 4 ; sub a1, a1, s7 }
    xm.bt s3, .vector_loop

.tail_end:
    { xm.shl s3, a3, 1 ; xm.brff a3, .done }
    { xm.mkmsk s3, s3 ; addi s4, a3, -1 }
    { li a3, 16 ; xm.bu .vector_loop_last }

.done:
    xm.lddsp s2, s3, 0
    xm.lddsp s4, s5, 8
    xm.lddsp s6, s7, 16
    xm.retsp (NSTACKWORDS * 4)
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
