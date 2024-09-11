#include <xs1.h>

    
// dequantize_int16_tensor_asm(int16_t *output, int16_t *input, int elements, char *blob);

#define FUNCTION_NAME dequantize_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     2

    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output array
    // r1: input array
    // r2: number of elements
    // r3: blob, floating point multiple and floating point addition
    //     0x40008000 afterwards (constant to make int16_t a float)
    // r4: floating point multipler
    // r5: floating point addition
    // r11: temp value
    .align 16

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop }
    std   r4, r5, sp[0]
    { ldw r4, r3[0]             ; ldap r11, magic_constant }
    { ldw r11, r11[0]           ; nop }
    { ldw r5, r3[1]             ; sub r2, r2, 1 }
dequantize_tensor_16_loop:
    ld16s r3, r1[r2]
    add r3, r3, r11          
    fadd r3, r3, r5
    fmul r3, r3, r4
    stw r3, r0[r2]            
    { bt  r2, dequantize_tensor_16_loop ; sub r2, r2, 1  }

    ldd   r4, r5, sp[0]
    retsp NSTACKWORDS
magic_constant:
    .word 0x40008000
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

    
