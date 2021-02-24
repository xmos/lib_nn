// Copyright 2020 XMOS LIMITED. This Software is subject to the terms of the 
// XMOS Public License: Version 1

#if defined(__XS3A__)

#include "nn_config.h"

/*


typedef struct {
    struct {
        int16_t shr;
        int16_t multiplier;
    } input[2];
    struct {
        int32_t bias;
        int8_t shr;
    } output;
} nn_add_params_t;


void add_elementwise(
    int8_t Y[],
    const int8_t X1[],
    const int8_t X2[],
    const nn_add_params_t* params,
    const unsigned output_start,
    const unsigned output_count);
*/

#ifndef NN_USE_REF
  #define FUNCTION_NAME add_elementwise
#else
  #define FUNCTION_NAME add_elementwise_asm
#endif // NN_USE_REF

#define NSTACKVECS      8
#define NSTACKWORDS     ((NSTACKVECS)*8 + 10)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define PARAMS_SHR1         0
#define PARAMS_IN1_MULT     1
#define PARAMS_SHR2         2
#define PARAMS_IN2_MULT     3
#define PARAMS_OUT_BIAS_LO  4
#define PARAMS_OUT_BIAS_HI  5
#define PARAMS_OUT_SHR_BYTE 12


#define STACK_VECS_START        (NSTACKWORDS - (NSTACKVECS * 8))
#define STACK_VEC_IN1           ((STACK_VECS_START)+0)
#define STACK_VEC_IN2           ((STACK_VECS_START)+8)
#define STACK_VECS_DP_START     ((STACK_VECS_START)+16)

#define DP_VEC_IN1_MULT     0
#define DP_VEC_IN2_MULT     8
#define DP_VEC_BIAS_HI      16
#define DP_VEC_BIAS_LO      24
#define DP_VEC_OUT_SHR      32

#define STACK_OUT_START     NSTACKWORDS+1
#define STACK_OUT_COUNT     NSTACKWORDS+2

#define STACK_ORIG_DP       8


#define Y           r0
#define X1          r1
#define X2          r2
#define N           r3
#define _32         r4
#define X1_shr      r5
#define X2_shr      r6
#define chans       r7
#define vec_tmp1    r8
#define vec_tmp2    r9
#define tmp         r10

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[1]
    std r6, r7, sp[2]
    std r8, r9, sp[3]
    
    // _32 = r4, this ldc is currently doing nothing because r4 is changed below.
{   ldc _32, 32                             ;   stw r10, sp[1]                          }

// Put dp on the stack and move sp[STACK_VECS_DP_START] into dp.
// The stack vectors we need will be stored relative to dp (can't do the same from sp)
{   ldaw dp, sp[STACK_VECS_DP_START]        ;   stw dp, sp[STACK_ORIG_DP]               }

#define params      r3

// Add the start offset to X1 and X2
    ldw tmp, sp[STACK_OUT_START]
{   add X1, X1, tmp                         ;   add X2, X2, tmp                         }

// Load X1_shr and X2_shr from the params 
// also apply start offset to Y
{   ldc r4, PARAMS_SHR1                     ;                                           }
{   ldc r4, PARAMS_SHR2                     ;   ld16s X1_shr, params[r4]                }
{   add Y, Y, tmp                           ;   ld16s X2_shr, params[r4]                }

// In order to use most of the params (X1_shr and X2_shr excluded), we need to duplicate each one
// into a full vector's worth.

// X1 Multiplier vector
{   ldc r4, PARAMS_IN1_MULT                 ;                                           }
{   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

// X2 Multiplier vector
{   ldc r4, PARAMS_IN2_MULT                 ;                                           }
{   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

// Low 16 bits of bias vector
{   ldc r4, PARAMS_OUT_BIAS_LO              ;                                           }
{   ldaw r11, dp[DP_VEC_BIAS_LO]            ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

// High 16 bits of bias vector
{   ldc r4, PARAMS_OUT_BIAS_HI              ;                                           }
{   ldaw r11, dp[DP_VEC_BIAS_HI]            ;   ld16s tmp, params[r4]                   }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

// Output Shr vector
{   ldc r4, PARAMS_OUT_SHR_BYTE             ;                                           }
{   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   ld8u tmp, params[r4]                    }
{   zext tmp, 16                            ;   shl chans, tmp, 16                      }
{   or tmp, tmp, chans                      ;   bl .L_std_func                          }

// Hop over the duplicating function
{                                           ;   bu .L_std_func_end                      }
#undef params

.L_std_func:
    std tmp, tmp, r11[0]
    std tmp, tmp, r11[1]
    std tmp, tmp, r11[2]
    std tmp, tmp, r11[3]
    retsp 0
.L_std_func_end:

// Load number of bytes to output from stack
    ldw N, sp[STACK_OUT_COUNT]
  
// Set VPU mode to 8-bit
{   ldaw vec_tmp1, sp[STACK_VEC_IN1]        ;   ldc _32, 32                             }
{   ldaw vec_tmp2, sp[STACK_VEC_IN2]        ;                                           }
{   shl r11, _32, 4                         ;                                           }
{                                           ;   vsetc r11                               }

// Exit early if there's no work to do.
{                                           ;   bf N, .L_loop_end                       }


/*
  Most of the instructions on the left side here are for determining the number of channels to output for that
  iteration. The loop can handle up to 16 output channels per iteration.

  Here's the logic:
    N = [remaining bytes to process]

    ldc tmp 15          
    lsu chans, tmp, N   
    shl chans, chans, 4  // i.e.   chans <-- (15 < N)? 16 : 0

    mov tmp, N           
    zext tmp, 4          // i.e.   tmp <-- N % 16

    shr tmp, tmp, chans   // i.e.  tmp <-- (15 < N)?  0 : (N % 16)
    add chans, tmp, chans // i.e.  chans <-- (15 < N)? 16 : (N % 16)
    
*/

.L_loop_top:

/*
    - X1 and X2 are 8-bit values, but we need them to be 16 bits. Doing a VLMACC in 8-bit mode (with vC=1) effectively
      expands them to 16 bits
        - I don't remember why I move them to vec_tmp1/vec_tmp2 before doing the VLMACC..
    - Apply X1_shr and X2_shr to X1 and X2, then store back on the stack
*/

    //  X1 and X2 are 8-bit values, but we need them to be 16 bits. Doing a VLMACC in 8-bit mode (with vC=1) effectively
    //  expands them to 16 bits. I don't recall why we don't start with the VLMACCs right away...
    {                                           ;   vldd X1[0]                              }
    {                                           ;   vstd vec_tmp1[0]                        }
    {                                           ;   vldd X2[0]                              }
    {                                           ;   vstd vec_tmp2[0]                        }
    {   ldc tmp, 15                             ;   vclrdr                                  }
        ldap r11, vpu_vects_vec_0x01
    {   lsu chans, tmp, N                       ;   vldc r11[0]                             }
    {   shl chans, chans, 4                     ;   vlmacc vec_tmp1[0]                      }
    {   mov tmp, N                              ;   vstr vec_tmp1[0]                        }
    {   zext tmp, 4                             ;   vclrdr                                  }
    {   shr tmp, tmp, chans                     ;   vlmacc vec_tmp2[0]                      }
    {   shl r11, _32, 3                         ;   vstr vec_tmp2[0]                        }
    // Switch VPU to 16-bit mode
    {   add chans, tmp, chans                   ;   vsetc r11                               }

    // Apply X1_shr and X2_shr to the X1 and X2 values
        vlashr vec_tmp1[0], X1_shr
    {   add X1, X1, chans                       ;   vstr vec_tmp1[0]                        }
        vlashr vec_tmp2[0], X2_shr
    {   ldaw r11, dp[DP_VEC_BIAS_HI]            ;   vstr vec_tmp2[0]                        }

    // If we're bypassing the symmetric saturation logic, store -128's in the output channels
#if !(CONFIG_SYMMETRIC_SATURATION_add_elementwise)
        ldaw r11, cp[vpu_vects_vec_0x80]
    {   mkmsk tmp, chans                        ;   vldr r11[0]                             }
        vstrpv Y[0], tmp
    {   ldaw r11, dp[DP_VEC_BIAS_HI]            ;                                           }
#endif // CONFIG_SYMMETRIC_SATURATION_add_elementwise

    // Load the output bias into vD:vR
    {   ldaw r11, dp[DP_VEC_BIAS_LO]            ;   vldd r11[0]                             }
    {   ldaw r11, dp[DP_VEC_IN1_MULT]           ;   vldr r11[0]                             }

    // MACC both inputs into vD:vR, using their respective multipliers.
    {   add X2, X2, chans                       ;   vldc r11[0]                             }
    {   ldaw r11, dp[DP_VEC_IN2_MULT]           ;   vlmacc vec_tmp1[0]                      }
    {   sub N, N, chans                         ;   vldc r11[0]                             }
    {   shl r11, _32, 4                         ;   vlmacc vec_tmp2[0]                      }

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise
    // If using symmetric saturation, switch to 8-bit mode and apply the output shift
    {   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   vsetc r11                               }
    {   mkmsk tmp, chans                        ;   vlsat r11[0]                            }
    // Store result in output
        vstrpv Y[0], tmp

#else 

    // If not using symmetric saturation logic, apply the output shift in 16-bit mode
    {   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   vstd vec_tmp2[0]                        }
    {                                           ;   vstr vec_tmp1[0]                        }
    {                                           ;   vlsat r11[0]                            }

    // VLSAT cleared vD[], but we'll need to do VLSAT on the same accumulators in 8-bit mode below, so pull the high
    // accumulator bits back into vD
    {                                           ;   vldd vec_tmp2[0]                        }

    // We've applied the output shift, so all values should be between -128 and 127 (inclusive). If we add
    // 127 (0x007F) then only the values that were -128 should still be negative.
        ldaw r11, cp[vpu_vects_vec_0x007F]
    {                                           ;   vladd r11[0]                            }

    // vdepth1 gives us a 16-bit bitmask, with bit k being set if vR[k] (16-bit) is negative. So, this mask will tell
    // us which elements saturate to -128.
    {   mkmsk tmp, chans                        ;   vdepth1                                 }
    {   mov r11, vec_tmp1                       ;   vstr vec_tmp2[0]                        }
    {                                           ;   ldw vec_tmp2, sp[STACK_VEC_IN2]         }

    // Reload the original accumulator value into vR  (now vD:vR have the original accumulators again)
    // ANDNOT applied to the mask we loaded because we already put -127 into the output buffer, so we actually need to
    // know which elements NOT to overwrite
    {   andnot tmp, vec_tmp2                    ;   vldr r11[0]                             }

    // Switch the VPU back to 8-bit mode
    {   shl r11, _32, 4                         ;                                           }
    {   ldaw r11, dp[DP_VEC_OUT_SHR]            ;   vsetc r11                               }

    // Apply the output shift in 8-bit mode
    {   ldaw vec_tmp2, sp[STACK_VEC_IN2]        ;   vlsat r11[0]                            }

    // Write outputs using our mask
        vstrpv Y[0], tmp

#endif // CONFIG_SYMMETRIC_SATURATION_add_elementwise

    // Advance Y by chans, and loop again if N isn't 0.
    {   add Y, Y, chans                         ;   bt N, .L_loop_top                       }

.L_loop_end:


    
.Lfunc_end:
// restore the old dp
{                                           ;   ldw dp, sp[STACK_ORIG_DP]               }
{                                           ;   ldw r10, sp[1]                          }
    ldd r4, r5, sp[1]
    ldd r6, r7, sp[2]
    ldd r8, r9, sp[3]
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



