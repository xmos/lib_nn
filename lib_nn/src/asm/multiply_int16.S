#include <xs1.h>

    
// multiply_int16_tensor_asm(int16_t *output, int16_t *input, int16_t *input2, int elements, char *blob);

#define FUNCTION_NAME multiply_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     8
#define ARGUMENT4       (NSTACKWORDS+1)
    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: input2
    // r3: number of elements,
    // r4: blob, then rounding value
    // r5, r6: temporary values
    // r7, r8: temporary values high/low accumulator
    // r9: shift - 16
    // r10: multiplier
    // r11: shift
    .align 16
    .skip  4
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]
    std   r8, r9, sp[2]
    stw   r10, sp[6]
    { bf    r3, skip              ; nop }
    { ldw   r4, sp[ARGUMENT4]     ; sub r3, r3, 1     }
    { ldw   r10, r4[0]            ; ldc  r5, 16       }
    { ldc   r4, 1                 ; shr  r11, r10, 16  }
    { sub   r9, r11, r5           ; sub    r5, r11, 1 }
    { shl   r4, r4, r5            ; sext  r10, 16 }
multiply_tensor_16_loop:
    { ld16s r5, r2[r3]            ; ldc r7, 0 }
    { ld16s r6, r1[r3]            ; add r8, r4, 0 }
    mul     r5, r5, r6
    maccs   r7, r8, r5, r10
    lsats   r7, r8, r9
    lextract r7, r7, r8, r11, 16
    st16    r7, r0[r3]
    { bt    r3, multiply_tensor_16_loop ; sub r3, r3, 1 }
skip:   

    ldd   r4, r5, sp[0]
    ldd   r6, r7, sp[1]
    ldd   r8, r9, sp[2]
    ldw r10, sp[6]
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




    

#undef FUNCTION_NAME
#undef NSTACKWORDS
#undef ARGUMENT4


    
// requantize_int16_tensor_asm(int16_t *output, int16_t *input, int elements, char *blob);

#define FUNCTION_NAME requantize_int16_tensor_asm

	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    .text
    
#define NSTACKWORDS     12
#define ARGUMENT4       (NSTACKWORDS+1)

    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: number of elements, then mask of number of final elements
    // r3: blob, 16 multipliers
    // r5: number of elements / 16
    // r6: temp vector
    // r7: 32
    // r11: temp.
    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 32    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]

    { shr   r5, r2, 4            ; zext  r2, 4     }       // Number of iterations/modulo
    { ldaw  r6, sp[4]            ; shl   r2, r2, 1 }
    { mkmsk r2, r2               ; shl   r11, r11, 3 }     // Make 0x100
    { ldc   r7, 32               ; vsetc r11       }

    ldaw r11, cp[eight_thousand]
    bu    requantise_tensor_16_loop_entry 
requantise_tensor_16_loop:
    { vstd     r0[0]              ; add r0, r0, r7 }
requantise_tensor_16_loop_entry:
    { vldd     r1[0]              ; nop }
    { vldr     r11[0]             ; nop }
    { vldc     r1[0]              ; add r1, r1, r7 }
    { vlmacc   r3[0]              ; nop            }
    { vlmacc   r3[0]              ; nop            }
    
    { bt       r5, requantise_tensor_16_loop ; sub   r5, r5, 1 }
    { vstd     r6[0]              ; add r11, r6, 0 }
    { vldr     r11[0]             ; nop }
    vstrpv   r0[0], r2

    ldd   r4, r5, sp[0]
    ldd   r6, r7, sp[1]

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
.Ltmp2:
    .size FUNCTION_NAME, .Ltmp2-FUNCTION_NAME

