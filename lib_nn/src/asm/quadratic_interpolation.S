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
    .globl quadratic_interpolation_128
    .globl quadratic_interpolation_128.nstackwords
    .linkset quadratic_interpolation_128.nstackwords, NSTACKWORDS
    .cc_top quadratic_interpolation_128.function,quadratic_interpolation_128

    .issue_mode dual
    .align 16
quadratic_interpolation_128:
    { dualentsp NSTACKWORDS     ; ldc r11, 0x1}
    std   r4, r5, sp[1] 
    std   r6, r7, sp[2]
    { shl r11, r11, 8           ; nop }
    std   r9, r10, sp[3]
    { stw   r8, sp[8]           ; nop }
    { vsetc   r11               ; ldaw r4, sp[10] }
    { shl   r5, r3, 4           ; nop }
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
    ldd   r9, r10, sp[0]
    ldd   r4, r5, sp[1] 
    ldd   r6, r7, sp[2] 
    ldw   r8, sp[6] 
    { retsp NSTACKWORDS           ; nop }

zero_offset_7_bit: 
    .word   0x00400040, 0x00400040, 0x00400040, 0x00400040
    .word   0x00400040, 0x00400040, 0x00400040, 0x00400040
zero_offset_9_bit: 
    .word   0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00
    .word   0xFF00FF00, 0xFF00FF00, 0xFF00FF00, 0xFF00FF00
    
    .cc_bottom quadratic_interpolation_128.function
