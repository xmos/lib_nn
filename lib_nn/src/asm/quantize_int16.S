#include <xs1.h>

    
// quantize_int16_tensor_asm(int16_t *output, int16_t *input, int elements, char *blob);

#define FUNCTION_NAME quantize_int16_tensor_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     6

    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    // r0: output array
    // r1: input array
    // r2: number of elements
    // r3: blob, floating point multiple and floating point addition
    //     0x8000 afterwards (constant to make signed unsigned)
    // r4/r5/r6: temp value
    // r7: floating point multipler
    // r8: maxint
    // r9: minint
    // r11: 8192.0
    .align 16
L0_5__MAXMIN:
    .word  0xBB800000         // -32768.0 * 2^-23
    .word  0x3B7FFE00         // +32767.0 * 2^-23
    .word  0x33800000         // +0.5 * 2^-23
    .align 16
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std   r4, r5, sp[0]
    std   r6, r7, sp[1]
    ldap r11, L0_5__MAXMIN
    std   r8, r9, sp[2]    
    ldd   r8, r9, r11[0]
    ldw   r11, r11[2]
    { ldw r7, r3[0]               ;  ldc r3, 0 }
    
quantize_tensor_16_loop:
    { ldw   r6, r1[0]             ; add r1, r1, 4 }
    fmacc r6, r11, r6, r7
    fgt   r4, r6, r8
    { bt    r4, maxint            ; nop }
    flt   r4, r6, r9
    { bt    r4, minint            ; nop }
    fsexp r4, r5, r6
    fmant r6, r6
    { bf  r4, fpos                ; sub r2, r2, 1 }
    { neg r6, r6                  ; nop }
fpos:   
    { shl r6, r6, r5              ; nop }
storeme:    
    st16 r6, r0[r3]
    { bt  r2, quantize_tensor_16_loop ; add r0, r0, 2  }

    ldd   r4, r5, sp[0]
    ldd   r6, r7, sp[1]
    ldd   r8, r9, sp[2]
    retsp NSTACKWORDS
    
maxint:
    ldc  r6, 0x7fff
    { bu  storeme                 ; sub r2, r2, 1 }
minint:
    ldc  r6, 0x8000
    { bu  storeme                 ; sub r2, r2, 1 }
    
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

    
