// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual


// extern void expand_8_to_16_ref_asm(int16_t *out, int8_t *in, int N);

#define FUNCTION_NAME expand_8_to_16_ref_asm

#define NSTACKWORDS  0
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4                      // loop starts with vclrdr - no align needed.
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; ldap r11, ones   }
    { vldc    r11[0]            ; ldc  r11, 0x20   }
    { shl     r11, r11, 4       ; nop              }
    { vsetc   r11               ; nop              }
    { zext    r2, 4             ; shr r11, r2, 4   }
    { bf      r11, tail         ; ldc r3, 16       }

loop:   
    { vclrdr                    ; sub r11, r11, 1  }
    { vlmacc  r1[0]             ; add r1, r1, r3   }
    { vstr    r0[0]             ; add r0, r0, r3   }
    { bt      r11, loop         ; add r0, r0, r3   }

tail:
    { bf      r2, done          ; nop              }
    { vclrdr                    ; shl r2, r2, 1    }
    { vlmacc  r1[0]             ; mkmsk r2, r2     }
    vstrpv    r0[0], r2
    
done:   
    retsp NSTACKWORDS

ones:
    .word 0x01010101
    .word 0x01010101
    .word 0x01010101
    .word 0x01010101
    
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

#endif // defined(__XS3A__)

#if (defined(__VX4A__) || defined(__VX4B__))

#define FUNCTION_NAME expand_8_to_16_ref_asm

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    li t3, 0x200
    { xm.ldap t3, ones ; xm.vsetc t3 }
    { li a3, 16 ; xm.vldc t3 }
    { xm.zext a2, 4 ; xm.shr t3, a2, 4 }
    xm.brff t3, .tail

.loop:
    { addi t3, t3, -1 ; xm.vclrdr }
    { add a1, a1, a3 ; xm.vlmacc0 a1 }
    { add a0, a0, a3 ; xm.vstr a0 }
    { add a0, a0, a3 ; xm.bt t3, .loop }

.tail:
    { xm.shl a2, a2, 1 ; xm.brff a2, .done }
    { xm.mkmsk a2, a2 ; xm.vclrdr }
    xm.vlmacc0 a1
    xm.vstrpv a0, a2

.done:
    ret

ones:
    .word 0x01010101
    .word 0x01010101
    .word 0x01010101
    .word 0x01010101

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  0
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif // defined(__VX4A__) || defined(__VX4B__)
