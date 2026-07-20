// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#include <xs1.h>

// Assume rhs is column major, [col, row]

/** 
 * void vect_mat_mul_int8_asm(
 *   const int8_t *lhs, 
 *   const int8_t *rhs, 
 *   int8_t *vpu_buffer, 
 *   uint32_t channel_size,     // lhs size, rhs row size
 *   uint32_t rhs_col_size);
 */

#define FUNCTION_NAME vect_mat_mul_int8_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
    
#define NSTACKWORDS     16
    .issue_mode dual

    .align 4

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

#define lhs         r0
#define rhs         r1
#define vpu_buffer  r2
#define ch_size     r3
#define ch_cnt      r6
#define col         r9

    ldw col, sp[NSTACKWORDS+1]
    ldc r11, 512
    { vsetc r11                 ; ldc r4, 32                    }
    { vclrdr                    ; shl r5, ch_size, 4            }   // r5: channel_size*16
    { vstr vpu_buffer[0]        ; add r11, vpu_buffer, r4       }   // make sure vpu buffer is all 0
    { vstr r11[0]               ; nop                           }   // make sure vpu buffer is all 0
    { add ch_cnt, ch_size, 0    ; ldc r11, 16                   }
    { lsu col, col, r11         ; sub rhs, rhs, ch_size         }   // rhs point to ch15
    { bf col, maccr_prepare     ; ldc col, 0                    }
    ldw col, sp[NSTACKWORDS+1]
    { sub col, r11, col         ; nop                           }   // col: tail loop cnt
    { shl col, col, 2           ; nop                           }   // col: maccr loop offset (in byte)

maccr_prepare:
    { lsu r11, ch_cnt, r4       ; nop                           }
    { bt r11, lhs_zero_padding  ; add r11, lhs, 0               }
    { vldc lhs[0]               ; add lhs, lhs, r4              }   // Load lhs to vC
    
maccr_loop_check:
    { bt col, tail_prepare      ; ldap r11, maccr               }

maccr:
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { vlmaccr rhs[0]            ; sub rhs, rhs, ch_size         }
    { add rhs, rhs, r5          ; lsu r11, ch_cnt, r4           } // rollback
    { add rhs, rhs, r4          ; bt r11, exit                  } // processed offset
    { sub ch_cnt, ch_cnt, r4    ; ldc r11, 16                   }  // channel_cnt -= 32
    { bt ch_cnt, maccr_prepare  ; nop                           }

exit:
    { vstd vpu_buffer[0]        ; add vpu_buffer, vpu_buffer, r4}
    { vstr vpu_buffer[0]        ; nop                           }
    ldw r4, sp[NSTACKWORDS-1]
    ldw r5, sp[NSTACKWORDS-2]
    ldw r6, sp[NSTACKWORDS-3]
    ldw r7, sp[NSTACKWORDS-4]
    ldw r8, sp[NSTACKWORDS-5]
    ldw r9, sp[NSTACKWORDS-6]
    retsp NSTACKWORDS

lhs_zero_padding:
    { vstr vpu_buffer[0]            ; add vpu_buffer, vpu_buffer, r4    }   // store acc vR to vpu buffer
    { vldr r11[0]                   ; mkmsk r11, ch_cnt                 }   // r11: mask of valid lhs byte                   }
    vstrpv vpu_buffer[0], r11   // store valid lhs data to vpu_buffer[63:32]
    { vldc vpu_buffer[0]            ; sub r11, vpu_buffer, r4           }   // r11: vpu_buffer[0]
    { vldr r11[0]                   ; add vpu_buffer, r11, 0            }   // restore vpu_buffer to vpu_buffer[0]
    { bu maccr_loop_check           ; nop                   }

tail_prepare:
    { add r11, r11, col             ; add r7, col, 0        }   // r7: tail loop count x4
maccr_tail:
    { vlmaccr lhs[0]                ; sub r7, r7, 4         }
    { bt r7, maccr_tail             ; sub rhs, rhs, ch_size }
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
