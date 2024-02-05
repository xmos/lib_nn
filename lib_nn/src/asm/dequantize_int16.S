#include <xs1.h>

    
// dequantize_int16_tensor_asm(int16_t *output, int16_t *input, int elements, char *blob);

#define FUNCTION_NAME dequantize_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     4

    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output array
    // r1: input array
    // r2: number of elements
    // r3: blob, floating point multiple and floating point addition
    //     0x8000 afterwards (constant to make signed unsigned)
    // r4: floating point multipler
    // r5: floating point addition
    // r6: temp value
    // r11: 8192.0
    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS      ; nop }
    ldc r11, 0x46                           // We can use the NOPs instead, but this aligns the loop.
    std   r4, r5, sp[0]
    { ldw r4, r3[0]             ; nop }
    { ldw r5, r3[1]             ; nop }
    ldc r3, 0x8000
    { stw   r6, sp[2]           ; sub r1, r1, r3 }
    { sub r1, r1, r3            ; shl r11, r11, 24 }
dequantize_tensor_16_loop:
    ld16s r6, r1[r3]
    { add r6, r6, r3              ; sub r2, r2, 1 }
    fmul r6, r6, r11
    fmacc r6, r5, r6, r4
    { stw r6, r0[0]               ; add r0, r0, 4 }
    { bt  r2, dequantize_tensor_16_loop ; add r1, r1, 2  }

    ldd   r4, r5, sp[0]
    ldw   r6, sp[2]
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

    
