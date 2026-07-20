// Copyright 2020-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__XS3A__)

/*
void add_asm(
    int8_t *y,
    int8_t *x1,
    int8_t *x2,
    nn_add_params_t *p,
    int output_start,
    int output_end);

typedef struct {
        int16_t m1[16];
        int16_t m2[16];
        int16_t shr[16];
        int16_t bias_hi[16];
        int16_t bias_lo[16];
} nn_add_params_t;
*/

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x, reg, val) { ldw reg, x[8]; nop} ; \
                                        { ldw reg, x[0]; nop} ; \
                                        { ldc reg, val ; nop }
#warning ddr fix in use
#else
#define OPTIONAL_DDR_SPILL(x, reg, val) /* */
#endif

#define FUNCTION_NAME add_elementwise_asm

#define NSTACKWORDS     (7 + 16 + 1) // 7 used + 16 for temp vecs + 1 to make even

.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define PARAMS_M1         0
#define PARAMS_M2         8
#define PARAMS_SHR        16
#define PARAMS_BIAS_HI    24
#define PARAMS_BIAS_LO    32

#define STACK_ORIG_DP     (NSTACKWORDS-17)
#define STACK_VEC1_TMP    (NSTACKWORDS-16)
#define STACK_VEC2_TMP    (NSTACKWORDS-8)
#define STACK_OUT_START   (NSTACKWORDS+1)
#define STACK_OUT_COUNT   (NSTACKWORDS+2)

#define arg_y_p           r0
#define arg_x1_p          r1
#define arg_x2_p          r2
#define arg_add_params_p  r3

#define tmp1              r4
#define tmp2              r5
#define counter           arg_add_params_p

#define bias_lo           r6
#define bias_hi           r7
#define m1                r8
#define m2                r9
#define shift             r10

.skip 8
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    // save stack
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    stw r10, sp[6]

    // Load vectors from params struct
    {stw dp, sp[STACK_ORIG_DP]        ;     ldc r11, 16}
    // Subtract 16 here from output pointer as an optmization
    // to add 16 before use in loop_main
    sub arg_y_p, arg_y_p, r11
    {set dp, arg_add_params_p         ;     shl r11, r11, 5}
    {ldaw bias_hi, dp[PARAMS_BIAS_HI] ;     ldaw bias_lo, dp[PARAMS_BIAS_LO]}
    {ldaw m1, dp[PARAMS_M1]           ;     ldaw m2, dp[PARAMS_M2]}
    // Load and init start position
    {ldaw shift, dp[PARAMS_SHR]       ;     ldw tmp1, sp[STACK_OUT_START]}
    // Set vector mode to 8 bit and set up loop counter
    {ldw tmp2, sp[STACK_OUT_COUNT]    ;     add arg_x1_p, arg_x1_p, tmp1}
    {vsetc r11                        ;     shr counter, tmp2, 4}
    {add arg_x2_p, arg_x2_p, tmp1     ;     add arg_y_p, arg_y_p, tmp1}

  label_loop_main:
    // x1 and x2 are 8 bit inputs
    // We load them and macc with vpu_vects_vec_0x01
    // to extend them into 16 bit
    vclrdr
    ldap r11, vpu_vects_vec_0x01
    {ldaw tmp1, sp[STACK_VEC1_TMP]    ;     ldaw tmp2, sp[STACK_VEC2_TMP]}
    {vldc r11[0]                      ;     ldc r11, 16}
    OPTIONAL_DDR_SPILL(arg_x1_p, r11, 16)
    {vlmacc arg_x1_p[0]               ;     add arg_x1_p, arg_x1_p, r11}
    {vstr tmp1[0]                     ;     add arg_y_p, arg_y_p, r11}
    vclrdr
    OPTIONAL_DDR_SPILL(arg_x2_p, r11, 16)
    {vlmacc arg_x2_p[0]               ;     add arg_x2_p, arg_x2_p, r11}
    // Load bias into acc
    // vldr - vR can only be loaded from r11
    {vstr tmp2[0]                     ;     mov r11, bias_lo}
    {vldr r11[0]                      ;     ldc r11, 32}
    // Set vector mode to 16 bit
    {vldd bias_hi[0]                  ;     shl r11, r11, 3}
    vsetc r11
    // Macc inputs with multipliers
    vldc tmp1[0]
    OPTIONAL_DDR_SPILL(m1, r11, 0)
    vlmacc m1[0]
    vldc tmp2[0]
    OPTIONAL_DDR_SPILL(m2, r11, 0)
    vlmacc m2[0]

    // Saturation fix
    // Save original 32-bit outputs
    // We saturate the number to 16-bits.
    // We then use the upper 16-bits to find out which numbers are
    // going to be -128.
    // -128's are written to output with a mask.
    // The lower 16-bits are later saturated using vlsat and
    // written with a mask to fill in remaining spaces.
    vstd tmp2[0]
    {vstr tmp1[0]                     ;     ldc tmp1, 32}
    OPTIONAL_DDR_SPILL(shift, r11, 0)
    {vlsat shift[0]                   ;     shl tmp1, tmp1, 4}
    vldd tmp2[0]
    ldap r11, vpu_vects_vec_0x007F
    {vladd r11[0]                     ;     mov r11, tmp1}
    vdepth1
    // Switch to 8-bit mode
    vsetc r11
#define saturated_mask    tmp2
#define mask              tmp1
    // Store saturated_mask and set default mask to 16 elements
    {vstr saturated_mask[0]           ;     mkmsk mask, 16}

    bt counter, label_store_with_mask
    // In last loop - make mask for remaining elements
    ldw tmp2, sp[STACK_OUT_COUNT]
    zext tmp2, 4
    mkmsk mask, tmp2

  label_store_with_mask:
    ldw saturated_mask, sp[STACK_VEC2_TMP]
    ldap r11, vpu_vects_vec_0x80
    and saturated_mask, saturated_mask, mask
    {vldr r11[0]                      ;     ldaw r11, sp[STACK_VEC1_TMP]}
    // Store -128's
    vstrpv arg_y_p[0], saturated_mask

    {vldr r11[0]                      ;     andnot mask, saturated_mask}
    OPTIONAL_DDR_SPILL(shift, r11, 0)
    vlsat shift[0]
    // Store other elements
    vstrpv arg_y_p[0], mask

    {sub counter, counter, 1          ;     bt counter, label_loop_main}

.Lfunc_end:
    // Restore stack
    ldw dp, sp[STACK_ORIG_DP]
    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    ldw r10, sp[6]
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

#endif

#if defined(__VX4B__)

/*
extern void add_elementwise_asm(
    int8_t y[], 
    const int8_t x1[],
    const int8_t x2[], 
    nn_add_params_t *params,
    const int output_start, 
    const int output_count);

typedef struct {
        int16_t m1[16];
        int16_t m2[16];
        int16_t shr[16];
        int16_t bias_hi[16];
        int16_t bias_lo[16];
} nn_add_params_t;
*/

#define FUNCTION_NAME add_elementwise_asm

#define PARAMS_M1         0
#define PARAMS_M2         32
#define PARAMS_SHR        64
#define PARAMS_BIAS_HI    96
#define PARAMS_BIAS_LO    128

#define STACK_X18         (0)
#define STACK_X19         (STACK_X18 + 4)
#define STACK_X20         (STACK_X19 + 4)
#define STACK_X21         (STACK_X20 + 4)
#define STACK_X22         (STACK_X21 + 4)
#define STACK_X23         (STACK_X22 + 4)
#define STACK_X24         (STACK_X23 + 4)
#define STACK_ORIG_DP     (STACK_X23 + 4)
#define STACK_VEC1_TMP    (STACK_ORIG_DP + 4)
#define STACK_VEC2_TMP    (STACK_VEC1_TMP + 32)
#define STACK_RA          (STACK_VEC2_TMP + 32)
#define NSTACKBYTES       (STACK_RA + 4)
#define ROUND_UP(X, Y)    (((X+Y-1)/Y)*Y)

#define arg_y_p           x10
#define arg_x1_p          x11
#define arg_x2_p          x12
#define arg_add_params_p  x13
#define arg_out_start     x14
#define arg_out_cnt       x15

#define tmp1              x18
#define tmp2              x19
#define counter           arg_add_params_p

#define bias_lo           x20
#define bias_hi           x21
#define m1                x22
#define m2                x23
#define shift             x24

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
.p2align 2
vpu_vects_vec_0x01:
	.word    0x01010101
	.word    0x01010101
	.word    0x01010101
	.word    0x01010101
vpu_vects_vec_0x007F:
    .word    0x007F007F
    .word    0x007F007F
    .word    0x007F007F
    .word    0x007F007F
    .word    0x007F007F
    .word    0x007F007F
    .word    0x007F007F
    .word    0x007F007F
vpu_vects_vec_0x80:
    .word    0x80808080
    .word    0x80808080
    .word    0x80808080
    .word    0x80808080
    .word    0x80808080
    .word    0x80808080
    .word    0x80808080
    .word    0x80808080

FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKBYTES, 16); xm.nop}
    xm.stdsp x18, x19, 0*8
    xm.stdsp x20, x21, 1*8
    xm.stdsp x22, x23, 2*8
	sw x24, 3*8(sp)

    // Load vectors from params struct
    // Subtract 16 here from output pointer as an optmization
    // to add 16 before use in loop_main
    { xm.ldcu x28, 16                   ; xm.nop                        }
    { sub arg_y_p, arg_y_p, x28         ; xm.shl x28, x28, 5            }
    // {set dp, arg_add_params_p         ;     shl r11, r11, 5}
    addi bias_hi, arg_add_params_p, PARAMS_BIAS_HI
    addi bias_lo, arg_add_params_p, PARAMS_BIAS_LO
    addi m1, arg_add_params_p, PARAMS_M1
    addi m2, arg_add_params_p, PARAMS_M2
    // Load and init start position
    addi shift, arg_add_params_p, PARAMS_SHR
    addi tmp1, arg_out_start, 0
    // Set vector mode to 8 bit and set up loop counter
    addi tmp2, arg_out_cnt, 0
    add arg_x1_p, arg_x1_p, tmp1
    { xm.vsetc x28                      ; xm.shri counter, tmp2, 4      }
    { add arg_x2_p, arg_x2_p, tmp1      ; add arg_y_p, arg_y_p, tmp1    }

  label_loop_main:
    // x1 and x2 are 8 bit inputs
    // We load them and macc with vpu_vects_vec_0x01
    // to extend them into 16 bit
    { xm.vclrdr                         ; xm.nop                        }
    lla x28, vpu_vects_vec_0x01
    addi tmp1, sp, STACK_VEC1_TMP
    addi tmp2, sp, STACK_VEC2_TMP
    { xm.vldc x28                       ; xm.ldcu x28, 16               }
    { xm.vlmacc0 arg_x1_p               ; add arg_x1_p, arg_x1_p, x28   }
    { xm.vstr tmp1                      ; add arg_y_p, arg_y_p, x28     }
    { xm.vclrdr                         ; xm.nop                        }
    { xm.vlmacc0 arg_x2_p               ; add arg_x2_p, arg_x2_p, x28   }
    // Load bias into acc
    // vldr - vR can only be loaded from x28
    { xm.vstr tmp2                      ; mv x28, bias_lo               }
    { xm.vldr x28                       ; xm.ldcu x28, 32               }
    // Set vector mode to 16 bit
    { xm.vldd bias_hi                   ; xm.shl x28, x28, 3            }
    { xm.vsetc x28                      ; xm.nop                        }
    // Macc inputs with multipliers
    xm.vldc tmp1
    xm.vlmacc0 m1
    xm.vlmacc1 m1
    xm.vldc tmp2
    xm.vlmacc0 m2
    xm.vlmacc1 m2

    // Saturation fix
    // Save original 32-bit outputs
    // We saturate the number to 16-bits.
    // We then use the upper 16-bits to find out which numbers are
    // going to be -128.
    // -128's are written to output with a mask.
    // The lower 16-bits are later saturated using vlsat and
    // written with a mask to fill in remaining spaces.
    xm.vstd tmp2
    { xm.vstr tmp1                      ; xm.ldcu tmp1, 32              }
    xm.vlsat shift                    
    { xm.vldd tmp2                      ; xm.shl tmp1, tmp1, 4          }
    
    lla x28, vpu_vects_vec_0x007F
    { xm.vladd x28                      ; mv x28, tmp1                  }
    xm.vdepth1
    // Switch to 8-bit mode
    xm.vsetc x28
#define saturated_mask    tmp2
#define mask              tmp1
    // Store saturated_mask and set default mask to 16 elements
    { xm.vstr saturated_mask            ; xm.mkmski mask, 16            }

    bnez counter, label_store_with_mask
    // In last loop - make mask for remaining elements
    addi tmp2, arg_out_cnt, 0
    xm.zexti tmp2, 4
    xm.mkmsk mask, tmp2

  label_store_with_mask:
    lw saturated_mask, STACK_VEC2_TMP(sp)
    lla x28, vpu_vects_vec_0x80
    and saturated_mask, saturated_mask, mask
    { xm.vldr x28                       ; addi x28, sp, STACK_VEC1_TMP  }
    // Store -128's
    xm.vstrpv arg_y_p, saturated_mask

    { xm.vldr x28                       ; xm.nop                        }
    { xm.andnot mask, saturated_mask    ; xm.nop                        }
    xm.vlsat shift
    // Store the result
    { xm.vstr x28                       ; xm.addi tmp2, x28, 0          }
    // Set to 16bit mode
    li x28, 256
    xm.vsetc x28
    li x28, -8
    xm.vlashr tmp2, x28
    xm.vdepth8
    // Store other elements
    xm.vstrpv arg_y_p, mask
    // Set to 8bit mode
    li x28, 512
    xm.vsetc x28

    addi x28, counter, 0
    addi counter, counter, -1
    bnez x28, label_loop_main

.Lfunc_end:
    // Restore stack
    xm.lddsp x18, x19, 0*8
    xm.lddsp x20, x21, 1*8
    xm.lddsp x22, x23, 2*8
	lw x24, 3*8(sp)

    {xm.retsp ROUND_UP(NSTACKBYTES, 16); xm.nop}

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKBYTES, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
