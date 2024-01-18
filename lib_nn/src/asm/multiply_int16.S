#include <xs1.h>

// output_transform_fn_int16_impl_asm(int16_t *output, int16_t *input, int16_t *blob, int elements);

#define FUNCTION_NAME multiply_int16_elementwise_constant_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     10
#define ARGUMENT4       (NSTACKWORDS+1)
    
    .align 4
    .issue_mode dual
    
    .globl FUNCTION_NAME

    .align 16
    .skip 4
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop    }
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
    .issue_mode  single

