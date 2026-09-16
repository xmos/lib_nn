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
    
#define NSTACKWORDS     8
    .issue_mode dual

    .align 4

    .globl FUNCTION_NAME
    .align 16
FUNCTION_NAME:
    { dualentsp NSTACKWORDS     ; nop    }

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]

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
    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    retsp NSTACKWORDS

lhs_zero_padding:
    { vstr vpu_buffer[0]            ; add vpu_buffer, vpu_buffer, r4    }   // store acc vR to vpu buffer
    { vldr r11[0]                   ; mkmsk r11, ch_cnt                 }   // r11: mask of valid lhs byte
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
    .set FUNCTION_NAME.nstackwords, NSTACKWORDS
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

#if defined(__VX4B__)

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

.p2align 2
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function

#define STACK_S2            (0)
#define STACK_S3            (STACK_S2 + 4)
#define STACK_S4            (STACK_S3 + 4)
#define STACK_S5            (STACK_S4 + 4)
#define STACK_S6            (STACK_S5 + 4)
#define STACK_S7            (STACK_S6 + 4)
#define STACK_RA            (STACK_S7 + 4)
#define NSTACKBYTES         (STACK_RA + 4)
#define ROUND_UP(X, Y)      (((X+Y-1)/Y)*Y)

#define lhs         a0
#define rhs         a1
#define vpu_buffer  a2
#define ch_size     a3
#define ch_cnt      s4
#define col         s7

FUNCTION_NAME:

    xm.entsp ROUND_UP(NSTACKBYTES, 16)

    xm.stdsp s2, s3, 0*8
    xm.stdsp s4, s5, 1*8
    xm.stdsp s6, s7, 2*8

    mv col, a4
    li t3, 512
    { xm.vsetc t3                   ; li s2, 32                 }
    { xm.vclrdr                     ; xm.shli s3, ch_size, 4    }   // s3: channel_size*16
    { xm.vstr vpu_buffer            ; add t3, vpu_buffer, s2    }   // make sure vpu buffer is all 0
    xm.vstr t3                                                      // make sure vpu buffer is all 0
    { add ch_cnt, ch_size, 0        ; li t3, 16                 }
    { sltu col, col, t3             ; sub rhs, rhs, ch_size     }   // rhs point to ch15
    { xm.brff col, maccr_prepare    ; li col, 0                 }
    mv col, a4
    sub col, t3, col                                                // col: tail loop cnt
    xm.shli col, col, 2                                             // col: maccr loop offset (in byte)

maccr_prepare:
    mv t3, lhs
    blt ch_cnt, s2, lhs_zero_padding
    { xm.vldc lhs                   ; add lhs, lhs, s2          }   // Load lhs to vC
    
maccr_loop_check:
    { xm.bt col, tail_prepare       ; xm.ldap t3, maccr         }

maccr:
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { xm.vlmaccr0 rhs               ; sub rhs, rhs, ch_size     }
    { add rhs, rhs, s3              ; sltu t3, ch_cnt, s2       }   // rollback
    { add rhs, rhs, s2              ; xm.bt t3, exit            }   // processed offset
    { sub ch_cnt, ch_cnt, s2        ; li t3, 16                 }   // channel_cnt -= 32
    bnez ch_cnt, maccr_prepare

exit:
    { xm.vstd vpu_buffer            ; add vpu_buffer, vpu_buffer, s2    }
    xm.vstr vpu_buffer
    xm.lddsp s2, s3, 0*8
    xm.lddsp s4, s5, 1*8
    xm.lddsp s6, s7, 2*8
    xm.retsp ROUND_UP(NSTACKBYTES, 16)

lhs_zero_padding:
    { xm.vstr vpu_buffer            ; add vpu_buffer, vpu_buffer, s2    }   // store acc vR to vpu buffer
    { xm.vldr t3                    ; xm.mkmsk t3, ch_cnt               }   // t3: mask of valid lhs byte
    xm.vstrpv vpu_buffer, t3    // store valid lhs data to vpu_buffer[63:32]
    { xm.vldc vpu_buffer            ; sub t3, vpu_buffer, s2            }   // t3: vpu_buffer[0]
    { xm.vldr t3                    ; mv vpu_buffer, t3                 }   // restore vpu_buffer to vpu_buffer[0]
    j maccr_loop_check

tail_prepare:
    { add t3, t3, col               ; mv s5, col                        }   // s5: tail loop count x4
maccr_tail:
    { xm.vlmaccr0 lhs               ; addi s5, s5, -4                   }
    { xm.bt s5, maccr_tail          ; sub rhs, rhs, ch_size             }
    jr t3

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKBYTES, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
