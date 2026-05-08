// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>

// scale_mul is in_scale / out_scale
// only support mean axis is the last axis, which end_dim_size = 1
// mean_dim_size needs to be word aligned

/** 
 * void mean_int8_asm(
 *      const int8_t *input,
 *      int8_t *output,
 *      const int start_dim_size,
 *      const int mean_dim_size,
 *      const int8_t *vpu_buffer,       // 64 byte
 *      const float in_zero_point_sum,  // in_zero_point*mean_dim_size
 *      const float out_zero_point,
 *      const float scale_mul) 
 */

#define FUNCTION_NAME mean_int8_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     32
    .issue_mode dual

    .align 4
    
filter:
    .word 0x01010101, 0x01010101, 0x01010101, 0x01010101
    .word 0x01010101, 0x01010101, 0x01010101, 0x01010101
    .word 0x00000000, 0x00000000, 0x00000000, 0x00000000
    .word 0x00000000, 0x00000000, 0x00000000, 0x00000000

#define EPV 32

    .globl FUNCTION_NAME
    .align 16
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop    }

    stw r4, sp[NSTACKWORDS-1]
    stw r5, sp[NSTACKWORDS-2]
    stw r6, sp[NSTACKWORDS-3]
    stw r7, sp[NSTACKWORDS-4]
    stw r8, sp[NSTACKWORDS-5]
    stw r9, sp[NSTACKWORDS-6]
    stw r10, sp[NSTACKWORDS-7]

#define input       r0
#define output      r1
#define channel_cnt r2
#define mean_size   r3
#define vpu_buffer  r4
#define mean_cnt    r5

    ldw vpu_buffer, sp[NSTACKWORDS+1]
    ldc r11, 512    // r11: VSETCTRL_TYPE_INT8
    { vsetc r11                     ; ldc r8, 16                    }   // r8: 16

prepare:
    { add mean_cnt, mean_size, 0    ; shl r9, mean_size, 4          }   // r9: 16th channel offset in byte
    { add input, input, r9          ; vclrdr                        }   // input: n+16th channel first byte
    { sub input, input, mean_size   ; nop                           }   // input: n+15th channel first byte

prepare_filter:
    { ldap r11, filter              ; ldc r10, 32                   }
    { lsu r9, mean_cnt, r10         ; nop                           }   // r9: mean_cnt < 32, r10: offset in byte
    { bf r9, maccr_prepare          ; nop                           }
    { sub r10, r10, mean_cnt        ; nop                           }
    { add r11, r11, r10             ; add r10, mean_cnt, 0          }   // r11: offset filter, r10: actual data goes through filter in next maccr

maccr_prepare:
    { vldc r11[0]                   ; nop                           }   // load filter to vC

maccr_loop_check:
    { lsu r11, channel_cnt, r8      ; sub r9, r8, channel_cnt       }   // r11: channel_cnt < 16, r9: tailing loop count
    { bt r11, maccr_tail_prepare    ; ldap r11, maccr_loop          }
maccr_loop:
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+15
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+14
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+13
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+12
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+11
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+10
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+9
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+8
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+7
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+6
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+5
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+4
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+3
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+2
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+1
    { vlmaccr input[0]              ; sub input, input, mean_size   }   // summation 32 byte in channel n+0
    { shl r9, mean_size, 4          ; add input, input, r10         }   // r9: number of byte to loop back, r0: +32 byte offset
    { add input, input, r9          ; ldc r10, 32                   }   // r0: next input for channel n+15, r10: 32
    { lsu r9, mean_cnt, r10         ; nop                           }   // r9: this is last loop
    { bt r9, output                 ; sub mean_cnt, mean_cnt, r10   }
    { bf mean_cnt, output           ; lsu r9, mean_cnt, r10         }   // r9: next is tailing loop
    { bt r9, prepare_filter         ; nop                           }
    { bu maccr_loop_check           ; nop                           }

output:
    { vstd vpu_buffer[0]            ; add r9, vpu_buffer, r10       }   // vpu_buffer[0]: vD, r9: vpu_buffer+32
    { vstr r9[0]                    ; ldc r9, 0                     }   // vpu_buffer[32]: vR, r9: 0
    std r3, r0, sp[0]
output_loop:
    std r2, r1, sp[1]           // store output_ptr and channel_cnt to sp
    { ldc r10, 16                   ; nop                           }   // r10: 16 as ld16s access in sub word
    { ld16s r7, vpu_buffer[r9]      ; add r9, r9, r10               }   // r7: vD[i+0...i+1]
    { ld16s r6, vpu_buffer[r9]      ; add r9, r9, 1                 }   // r6: vR[i+0...i+1]
    // prepare full 32-bit accumulator value
    { shl r7, r7, 16                ; shl r6, r6, 16                }
    { shr r6, r6, 16                ; ldc r1, 0                     }   // r1: exp=0
    { or r0, r7, r6                 ; ldc r6, 0                     }   // r0: 32bit accumulator, r6: 0
    bl s32_to_f32               // r0: accumulator value in float
    { ldw r1, sp[NSTACKWORDS+2]     ; sub r9, r9, r10               }   // r1: in zero point sum, r9: rollback for vD
    { ldw r2, sp[NSTACKWORDS+3]     ; eq r7, r9, r10                }   // r2: out zero point, r7: last output loop
    { ldw r3, sp[NSTACKWORDS+4]     ; nop                           }   // r3: scale
    fsub r0, r0, r1             // r0: acc-in_zero_sum
    fmacc r0, r2, r0, r3        // r0: r0*scale+out_zero
    bl round8                   // r0: round(r0) in int8_t
    ldd r2, r1, sp[1]           // r1: output_ptr, r2: channel_cnt
    st8 r0, r1[r6]              // store result to output
    { add r1, r1, 1                 ; sub r2, r2, 1                 }   // r2: num of channel out left
    { bf r2, exit                   ; nop                           }
    { bf r7, output_loop            ; nop                           }
    ldd r3, r0, sp[0]
    bu prepare

exit:
    ldw r4, sp[NSTACKWORDS-1]
    ldw r5, sp[NSTACKWORDS-2]
    ldw r6, sp[NSTACKWORDS-3]
    ldw r7, sp[NSTACKWORDS-4]
    ldw r8, sp[NSTACKWORDS-5]
    ldw r9, sp[NSTACKWORDS-6]
    ldw r10, sp[NSTACKWORDS-7]
    retsp NSTACKWORDS

maccr_tail_prepare:
    mul r7, r9, mean_size   // r7: input loopback to valid addr offset size (in byte)
    { shl r9, r9, 2                 ; nop                   }   // r9: maccr_loop offset (in byte)
    { add r11, r11, r9              ; sub input, input, r7  }   // r11: maccr_loop offset start addr
maccr_tail_loop:
    { vlmaccr input[0]              ; sub r9, r9, 4         }
    { bt r9, maccr_tail_loop        ; nop                   }
    { bau r11                       ; nop                   }

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords, NSTACKWORDS+s32_to_f32.nstackwords+round8.nstackwords
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
