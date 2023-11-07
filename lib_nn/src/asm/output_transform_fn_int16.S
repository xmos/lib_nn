#include <xs1.h>

// output_transform_fn_int16_impl_asm(int32_t *vDvR, int32_t *mul_add, int32_t *output, int elements);

#define FUNCTION_NAME output_transform_fn_int16_impl_asm

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
    { dualentsp NSTACKWORDS     ; add r11, r0, 0 }
    std   r4, r5, sp[1]
    { stw   r10, sp[1]          ; shl r10, r3, 1 }
    std   r6, r7, sp[2]
    std   r8, r9, sp[3]
    { mkmsk r5, 16              ; ldc r3, 0 }
    zip r5, r3, 1
    stw   r10, sp[0]
    { ldc r0, 4                 ; mkmsk r10, r10 }
    
output_transform_16_unzip_loop:   
    ldd   r6, r8, r11[0]          // vD first two words
    ldd   r7, r9, r11[4]          // vR first two words
    unzip r6, r7, 4
    unzip r8, r9, 4             // r6, r7, r8, r9 first four words
    std   r6, r7, r11[4]
    std   r8, r9, r11[0]
    { add   r11, r11, 8      ; sub r0, r0, 1 }
    bt    r0, output_transform_16_unzip_loop

    { ldc    r9, 32        ; ldc r8, 16      }  // Make -16 in r8
    { vldr   r11[0]        ; neg r8, r8      }
    { vlmul  r1[0]         ; add r1, r1, r9  }
    { vladd  r1[0]         ; add r1, r1, r9  }
    { vstr   r11[0]        ; and r5, r10, r3 }  
    vlashr r11[0], r8                           // Saturate to 0x800000001 / 0x7fffffff
    vstrpv r2[0], r5
    { sub    r11, r11, r9  ; shr r3, r3, 2   }
    { vldr   r11[0]        ; sub r0, r0, 1   }
    { vlmul  r1[0]         ; add r1, r1, r9  }
    { vladd  r1[0]         ; add r1, r1, r9  }
    { vstr   r11[0]        ; neg r7, r8      }  // Make +16 in r7
    vlashr r11[0], r8                           // Saturate to 0x800000001 / 0x7fffffff
    { vstr   r11[0]        ; and r5, r10, r3 }  // make 0x3333
    vlashr r11[0], r7                           // Move to lower half of word
    vstrpv r2[0], r5
    
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

