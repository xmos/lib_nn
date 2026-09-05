// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

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
    
#define NSTACKWORDS     16
    .issue_mode dual

    .align 4
    
filter:
    .word 0x01010101, 0x01010101, 0x01010101, 0x01010101
    .word 0x01010101, 0x01010101, 0x01010101, 0x01010101
    .word 0x00000000, 0x00000000, 0x00000000, 0x00000000
    .word 0x00000000, 0x00000000, 0x00000000, 0x00000000

    .globl FUNCTION_NAME
    .align 16
FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[2]
    std r6, r7, sp[3]
    std r8, r9, sp[4]
    stw r10, sp[10]

#define input       r0
#define OUTPUT      r1
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
    { shr r6, r6, 16                ; ldc r1, 23                    }   // r1: 23
    { or r0, r7, r6                 ; ldc r6, 0                     }   // r0: 32bit accumulator, r6: 0
    // Convert int32 to float
    ashr r2, r0, 32 // r2: sign
    bf r2, int32_to_f32_pack_not_neg
    neg r0, r0
    int32_to_f32_pack_not_neg:
        fmake r0, r2, r1, r6, r0
    // Convert int32 to float done
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
    ldd r4, r5, sp[2]
    ldd r6, r7, sp[3]
    ldd r8, r9, sp[4]
    ldw r10, sp[10]
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
    .set FUNCTION_NAME.nstackwords, NSTACKWORDS + round8.nstackwords
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

// scale_mul is in_scale / out_scale
// only support mean axis is the last axis, which end_dim_size = 1

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

.p2align 2
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function

.p2align 2
filter:
    .word 0x01010101, 0x01010101, 0x01010101, 0x01010101
    .word 0x01010101, 0x01010101, 0x01010101, 0x01010101
    .word 0x00000000, 0x00000000, 0x00000000, 0x00000000
    .word 0x00000000, 0x00000000, 0x00000000, 0x00000000

#define STACK_S2            (0)
#define STACK_S3            (STACK_S2 + 4)
#define STACK_S4            (STACK_S3 + 4)
#define STACK_S5            (STACK_S4 + 4)
#define STACK_S6            (STACK_S5 + 4)
#define STACK_S7            (STACK_S6 + 4)
#define STACK_S8            (STACK_S7 + 4)
#define STACK_ALIGN_PAD     (STACK_S8 + 4)
#define STACK_A0_TMP        (STACK_ALIGN_PAD + 4)
#define STACK_A3_TMP        (STACK_A0_TMP + 4)
#define STACK_A1_TMP        (STACK_A3_TMP + 4)
#define STACK_A2_TMP        (STACK_A1_TMP + 4)
#define STACK_RA            (STACK_A2_TMP + 4)
#define NSTACKBYTES         (STACK_RA + 4)
#define ROUND_UP(X, Y)      (((X+Y-1)/Y)*Y)

#define input       a0
#define OUTPUT      a1
#define channel_cnt a2
#define mean_size   a3
#define vpu_buffer  s2
#define mean_cnt    s3

FUNCTION_NAME:
    xm.entsp ROUND_UP(NSTACKBYTES, 16)

    xm.stdsp s2, s3, 0*8
    xm.stdsp s4, s5, 1*8
    xm.stdsp s6, s7, 2*8
    sw s8, 3*8(sp)

    mv vpu_buffer, a4
    li t3, 512    // t3: VSETCTRL_TYPE_INT8
    { xm.vsetc t3                   ; li s6, 16                     }   // s6: 16

prepare:
    { add mean_cnt, mean_size, 0    ; xm.shli s7, mean_size, 4      }   // s7: 16th channel offset in byte
    { add input, input, s7          ; xm.vclrdr                     }   // input: n+16th channel first byte
    { sub input, input, mean_size   ; nop                           }   // input: n+15th channel first byte

prepare_filter:
    { xm.ldap t3, filter            ; li s8, 32                     }
    { sltu s7, mean_cnt, s8         ; nop                           }   // s7: mean_cnt < 32, s8: offset in byte
    { xm.brff s7, maccr_prepare     ; nop                           }
    { sub s8, s8, mean_cnt          ; nop                           }
    { add t3, t3, s8                ; add s8, mean_cnt, 0           }   // t3: offset filter, s8: actual data goes through filter in next maccr

maccr_prepare:
    { xm.vldc t3                    ; nop                           }   // load filter to vC

maccr_loop_check:
    { xm.ldap t3, maccr_loop        ; sub s7, s6, channel_cnt       }   // t3: maccr_loop, s7: tailing loop count
    bltu channel_cnt, s6, maccr_tail_prepare    // channel_cnt < s6 => maccr_tail_prepare

maccr_loop:
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+15
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+14
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+13
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+12
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+11
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+10
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+9
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+8
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+7
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+6
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+5
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+4
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+3
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+2
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+1
    { xm.vlmaccr0 input             ; sub input, input, mean_size   }   // summation 32 byte in channel n+0
    { xm.shli s7, mean_size, 4      ; add input, input, s8          }   // s7: number of byte to loop back, a0: +32 byte offset
    { add input, input, s7          ; li s8, 32                     }   // a0: next input for channel n+15, s8: 32
    bltu mean_cnt, s8, output           // mean_cnt < 32 => this is last loop
    sub mean_cnt, mean_cnt, s8          // mean_cnt -= 32
    xm.brff mean_cnt, output            // mean_cnt == 0 => this is last loop
    bltu mean_cnt, s8, prepare_filter   // mean_cnt < s8 => next is tailing loop
    j maccr_loop_check

output:
    { xm.vstd vpu_buffer            ; add s7, vpu_buffer, s8        }   // vpu_buffer[0]: vD, s7: vpu_buffer+32
    { xm.vstr s7                    ; li s7, 0                      }   // vpu_buffer[32]: vR, s7: 0
    xm.stdsp a3, a0, STACK_A0_TMP
output_loop:
    xm.stdsp a2, a1, STACK_A1_TMP   // store output_ptr and channel_cnt to sp
    li s8, 16                       // s8: 16 as ld16s access in sub word
    xm.ld16s s5, s7(vpu_buffer)     // s5: vD[i+0...i+1]
    add s7, s7, s8   
    xm.ld16s s4, s7(vpu_buffer)     // s4: vR[i+0...i+1]
    add s7, s7, 1
    // prepare full 32-bit accumulator value
    { xm.shli s5, s5, 16            ; xm.shli s4, s4, 16            }
    { srli s4, s4, 16               ; li a1, 23                     }   // a1: 23
    { or a0, s5, s4                 ; li s4, 0                      }   // a0: 32bit accumulator, s4: 0
    // Convert int32 to float
    srai a2, a0, 16
    srai a2, a2, 16 // a2: sign
    xm.brff a2, int32_to_f32_pack_not_neg
    neg a0, a0
    int32_to_f32_pack_not_neg:
        xm.fmake a0, a2, a1, s4, a0
    // Convert int32 to float done
    sub s7, s7, s8                  // s7: rollback for vD
    xm.eq s5, s7, s8                // a2: out zero point, s5: last output loop
    mv a1, a5                       // a1: in zero point sum
    mv a2, a6                       // a2: out zero point
    mv a3, a7                       // a3: scale
    xm.fsub a0, a0, a1              // a0: acc-in_zero_sum
    xm.fmacc a0, a2, a0, a3         // a0: a0*scale+out_zero
    call round8                     // a0: round(a0) in int8_t
    xm.lddsp a2, a1, STACK_A1_TMP   // a1: output_ptr, a2: channel_cnt
    xm.st8 a0, s4(a1)               // store result to output
    { add a1, a1, 1                 ; addi a2, a2, -1               }   // a2: num of channel out left
    beqz a2, exit
    beqz s5, output_loop
    xm.lddsp a3, a0, STACK_A0_TMP
    j prepare

exit:
    xm.lddsp s2, s3, 0*8
    xm.lddsp s4, s5, 1*8
    xm.lddsp s6, s7, 2*8
    lw s8, 3*8(sp)
    xm.retsp ROUND_UP(NSTACKBYTES, 16)

maccr_tail_prepare:
    mul s5, s7, mean_size   // s5: input loopback to valid addr offset size (in byte)
    xm.shli s7, s7, 2       // s7: maccr_loop offset (in byte)
    { add t3, t3, s7                ; sub input, input, s5          }   // t3: maccr_loop offset start addr
maccr_tail_loop:
    { xm.vlmaccr0 input             ; addi s7, s7, -4               }
    xm.bt s7, maccr_tail_loop
    jr t3

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKBYTES, 16)
.resource_list_add   FUNCTION_NAME, "callees", round8
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif