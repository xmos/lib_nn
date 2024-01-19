#include <xs1.h>

#if 0
DEPRECATED
// multiply_int16_constant_asm(int16_t *output, int16_t *input, int16_t *blob, int elements);

#define FUNCTION_NAME multiply_int16_constant_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     10
#define ARGUMENT4       (NSTACKWORDS+1)
    
    .issue_mode dual
    
    .globl FUNCTION_NAME

    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 32    }
    ldc   r11, 256
    vsetc r11
    std   r4, r5, sp[1]
    std   r6, r7, sp[2]
    std   r8, r9, sp[3]             // TODO: trim down saved regs and optimise dual issue
    stw   r10, sp[0]

    shr   r5, r3, 4                 // Number of iterations
    zext  r3, 4
    shl   r3, r3, 1
    mkmsk r3, r3
    ldc   r4, 32
    { ldw   r6, r2[0]             ; add r2, r2, 4 }
    bu    multiply_16_loop_entry 
multiply_16_loop:
    { vstr     r0[0]              ; add r0, r0, r4 }
multiply_16_loop_entry:
    vclrdr
    { vldc     r1[0]              ; add r1, r1, r4 }
    { vlmacc   r2[0]              ; add r2, r2, r4 }
    { vlsat    r2[0]              ; add r2, r2, r6 }
    
    { bt    r5, multiply_16_loop ; sub   r5, r5, 1 }
    vstrpv   r0[0], r3

    ldw   r10, sp[1]
    ldd   r4, r5, sp[1]
    ldd   r6, r7, sp[2]
    ldd   r8, r9, sp[3]

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
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME


#undef FUNCTION_NAME
#undef NSTACKWORDS
#undef ARGUMENT4

#endif
    
    
// multiply_int16_tensor_asm(int16_t *output, int16_t *input, int16_t *input2, int elements, char *blob);

#define FUNCTION_NAME multiply_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     12
#define ARGUMENT4       (NSTACKWORDS+1)
    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: input2
    // r3: number of elements, then mask of number of final elements
    // r4: blob, 16 multipliers
    // r5: number of elements / 16
    // r6: temp vector
    // r7: 32
    // r11: temp.
    .align 16
    .skip 12
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 32    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]

    { shr   r5, r3, 4            ; zext  r3, 4     }       // Number of iterations/modulo
    { ldaw  r6, sp[4]            ; shl   r3, r3, 1 }
    { ldw   r4, sp[ARGUMENT4]    ; shl   r11, r11, 3 }     // Make 0x100
    { mkmsk r3, r3               ; vsetc r11       }

    ldaw  r11, cp[eight_thousand]
    { ldc   r7, 32               ; bu    multiply_tensor_16_loop_entry  }
multiply_tensor_16_loop:
    { vstd     r0[0]              ; add r0, r0, r7 }
multiply_tensor_16_loop_entry:
    vclrdr
    { vldr     r11[0]             ; nop }
    { vldc     r1[0]              ; add r1, r1, r7 }
    { vlmacc   r2[0]              ; nop            }
    { vlmacc   r2[0]              ; add r2, r2, r7 }
    { vstd     r6[0]              ; nop            }
    vclrdr
    { vldr     r11[0]             ; nop }
    { vldc     r4[0]              ; nop            }
    { vlmacc   r6[0]              ; nop            }
    { vlmacc   r6[0]              ; nop            }
    { vlmacc   r6[0]              ; nop            }
    { vlmacc   r6[0]              ; nop            }
    
    { bt    r5, multiply_tensor_16_loop ; sub   r5, r5, 1 }
    { vstd     r6[0]              ; add r11, r6, 0 }
    { vldr     r11[0]             ; nop }
    vstrpv   r0[0], r3

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
.Ltmp1:
    .size FUNCTION_NAME, .Ltmp1-FUNCTION_NAME




    

#undef FUNCTION_NAME
#undef NSTACKWORDS
#undef ARGUMENT4


    
// quantise_int16_tensor_asm(int16_t *output, int16_t *input, char *blob, int elements);

#define FUNCTION_NAME quantise_int16_tensor_asm

	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    .text
    
#define NSTACKWORDS     12
#define ARGUMENT4       (NSTACKWORDS+1)

    .globl FUNCTION_NAME
    // r0: output
    // r1: input1
    // r2: blob, 16 multipliers
    // r3: number of elements, then mask of number of final elements
    // r5: number of elements / 16
    // r6: temp vector
    // r7: 32
    // r11: temp.
    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldc r11, 32    }
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]

    { shr   r5, r3, 4            ; zext  r3, 4     }       // Number of iterations/modulo
    { ldaw  r6, sp[4]            ; shl   r3, r3, 1 }
    { mkmsk r3, r3               ; shl   r11, r11, 3 }     // Make 0x100
    { ldc   r7, 32               ; vsetc r11       }

    ldaw r11, cp[eight_thousand]
    bu    quantise_tensor_16_loop_entry 
quantise_tensor_16_loop:
    { vstd     r0[0]              ; add r0, r0, r7 }
quantise_tensor_16_loop_entry:
    { vldd     r1[0]              ; nop }
    { vldr     r11[0]             ; nop }
    { vldc     r1[0]              ; add r1, r1, r7 }
    { vlmacc   r2[0]              ; nop            }
    { vlmacc   r2[0]              ; nop            }
    
    { bt       r5, quantise_tensor_16_loop ; sub   r5, r5, 1 }
    { vstd     r6[0]              ; add r11, r6, 0 }
    { vldr     r11[0]             ; nop }
    vstrpv   r0[0], r3

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

