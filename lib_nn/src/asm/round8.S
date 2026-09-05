// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>

/** 
 * int8_t round8(float r)     // round-half-away-from-zero
 */

#define FUNCTION_NAME round8

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     0
    .issue_mode dual

    .align 4
    
    .globl FUNCTION_NAME
    .align 16
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop                   }

    ldc r1, 0x42fe
    ldc r2, 0xc300
    { shl r1, r1, 16            ; shl r2, r2, 16        }   // r1: 127.0, r2: -128.0
    flt r1, r0, r1  // r1: r < 127.0
    { bf r1, overflow127        ; ldc r3, 22            }   // r3: 22
    fgt r2, r0, r2  // r2: r > -128.0
    { bf r2, underflow_neg_128  ; add r2, r0, 0         }   // r2: r

    fmant r0, r2
    fsexp r1, r2, r2    // r1: sign, r2: exp, r0: mantissa

    { sub r2, r3, r2            ; nop               }   // r2: 22-exp
    { shr r0, r0, r2            ; ldc r3, 1         }   // r0: %.1f(r)
    { and r3, r0, r3            ; shr r0, r0, 1     }   // r3: rounding bit, r0: integer_part
    { add r0, r0, r3            ; bf r1, exit       }   // r0: rounded integer
    { neg r0, r0                ; retsp NSTACKWORDS }

overflow127:
    ldc r0, 127
    retsp NSTACKWORDS

underflow_neg_128:
    ldc r0, 128
    neg r0, r0          // r0: -128
exit:
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

#endif

#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME round8

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    xm.entsp 0

    li a1, 0x42fe
    li a2, 0xc300
    { xm.shli a1, a1, 16            ; xm.shli a2, a2, 16    }   // a1: 127.0, a2: -128.0
    xm.flt a1, a0, a1                                           // a1: r < 127.0
    { xm.brff a1, overflow127       ; li a3, 22             }   // a3: 22
    xm.fgt a2, a0, a2                                           // a2: r > -128.0
    { xm.brff a2, underflow_neg_128 ; addi a2, a0, 0        }   // a2: r

    xm.fmant a0, a2
    xm.fsexp a1, a2, a2 // a1: sign, a2: exp, a0: mantissa

    sub a2, a3, a2                                      // a2: 22-exp
    { xm.shr a0, a0, a2         ; li a3, 1          }   // a0: %.1f(r)
    { and a3, a3, a0            ; srli a0, a0, 1    }   // a3: rounding bit, a0: integer_part
    { add a0, a0, a3            ; xm.brff a1, exit  }   // a0: rounded integer
    neg a0, a0
    xm.retsp 0

overflow127:
    li a0, 127
    xm.retsp 0

underflow_neg_128:
    li a0, 128
    neg a0, a0          // a0: -128
exit:
    xm.retsp 0
    
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  0
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif