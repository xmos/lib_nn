#include <xs1.h>

    
// add_int16_tensor_asm(int16_t *output, int16_t *input, int16_t *input2, int elements, char *blob);

#define FUNCTION_NAME add_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     24
#define ARGUMENT4       (NSTACKWORDS+1)
    .issue_mode dual

    .align 4
shifts:
    .word 0x000E000E, 0x000E000E, 0x000E000E, 0x000E000E
    .word 0x000E000E, 0x000E000E, 0x000E000E, 0x000E000E
minints:    
    .word 0x80008000, 0x80008000, 0x80008000, 0x80008000
    .word 0x80008000, 0x80008000, 0x80008000, 0x80008000
minintmults:    
    .word 0x40004000, 0x40004000, 0x40004000, 0x40004000
    .word 0x40004000, 0x40004000, 0x40004000, 0x40004000

    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: input2
    // r3: number of elements
    //     then mask of number of final elements
    // r4: blob, 16 multipliers
    // r5: number of elements / 16
    // r6: temp vector
    // r7: 32
    // r11: temp.

    // Overflow condition:
    //    accumulator gets to 0xE000 0000 or lower
    //    Add 0x2000 0000 and any negative value indicates overflow
    //    0x2000 0000 is 0x4000 x 0x4000
    .align 16
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 32    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]
    std   r8, r9, sp[2]
    stw   r10, sp[6]

    { shr   r5, r3, 4            ; zext  r3, 4     }       // Number of iterations/modulo
    { ldaw  r6, sp[8]            ; shl   r3, r3, 1 }
    { ldw   r4, sp[ARGUMENT4]    ; shl   r11, r11, 3 }     // Make 0x100
    { mkmsk r3, r3               ; vsetc r11       }
    { ldap  r11, shifts          ; ldc   r7, 32     }
    { ldap  r11, minintmults     ; add   r8, r11, 0 }
    { stw   r3, sp[7]            ; add   r9, r11, 0 }
    { bf    r5, zero_iterations  ; nop }
    mkmsk r3, 32
add_tensor_16_loop:
    { vclrdr                      ; nop }
    // First calculate m0 * i0[] + m1 * i1[]
    { vldc     r4[0]              ; add r4, r4, r7 }
    { vlmacc   r1[0]              ; add r1, r1, r7 }
    { vldc     r4[0]              ; sub r4, r4, r7 }
    { vlmacc   r2[0]              ; add r2, r2, r7 }
    // Now save the accumulator on the stack
    { vstd     r6[0]              ; add r11, r6, r7 }
    { vstr     r11[0]             ; nop            }
    // Now shift down by 14 bits to get the anwer - 0x8001 as underflow
    { vlsat    r8[0]              ; nop            }
    // Save this as the answer
    vstrpv     r0[0], r3
    // Restore teh accumulator with the full precision product
    { vldr     r11[0]             ; nop            }
    { vldd     r6[0]              ; nop            }
    // Now add 0x4000 x 0x4000 to the accumulator
    // This makes all values positive except < 0x8000.0000
    { vldc     r9[0]              ; nop            }
    { vlmacc   r9[0]              ; nop            }
    // Shift down, and keep sign bits
    { vlsat    r8[0]              ; sub   r5, r5, 1 }
    vsign
    vdepth1
    { vstr     r6[0]              ; ldap r11, minints}
    // Grab the sign bits, and zip them together once to create two bits for each wrong 16-bit
    ldw        r10, r6[0]
    { vldr     r11[0]             ; add        r11, r10, 0 }
    zip        r11, r10, 0
    and        r10, r10, r3
    // Overwrite minints
    vstrpv     r0[0], r10
    { bt    r5, add_tensor_16_loop ; add r0, r0, r7 }
    ldw     r3, sp[7]
    stw     r5, sp[7] // stop next iteration
zero_iterations:    
    { bf    r3, done               ; nop }
    { bu add_tensor_16_loop       ; ldc r5, 1 }

done:   
    ldd   r4, r5, sp[0]
    ldd   r6, r7, sp[1]
    ldd   r8, r9, sp[2]
    ldw   r10, sp[6]

    retsp NSTACKWORDS

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp1:
    .size FUNCTION_NAME, .Ltmp1-FUNCTION_NAME



