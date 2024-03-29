// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__XS3A__)
#include "nn_config.h"
#include "xs3_vpu.h"

/*
void bsign_8(
    bnn_b32_t* Y,
    const int8_t* X,
    const int8_t* zero_point_vect,
    const nn_bsign_8_job_t* job);
*/

#ifndef NN_USE_REF
  #define FUNCTION_NAME bsign_8
#else
  #define FUNCTION_NAME bsign_8_asm
#endif // NN_USE_REF

    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define arg_Y           r0
#define arg_X           r1
#define arg_zero_pv     r2
#define arg_job         r3

#define Y               arg_Y
#define X               arg_X
#define length          r2
#define tail_mask       r3
#define tmp             r4
#define _32             r5


#define zero_pv         r11


#ifdef USE_DDR_FIX
#define NSTACKWORDS     (3)
#define ddr_spill       r6
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ;\
                              { ldw ddr_spill, x[0]; nop}
#else
#define NSTACKWORDS     (2)
#define OPTIONAL_DDR_SPILL(x) /* */
#endif


#define JOB_START       0
#define JOB_LENGTH      1

#define VSR_VALUE       XS1_VSR_TYPE_SET(0, XS1_VSETC_TYPE_INT8)

.skip 8
FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
#ifdef USE_DDR_FIX
    {stw ddr_spill, sp[2]                       ;   nop                                    }
#endif

    {   ldw tmp, arg_job[JOB_START]             ;   ldc _32, 32                             }
    {   add X, X, tmp                           ;                                           }
    {   ldc r11, VSR_VALUE                      }
    {   shr tmp, tmp, 3                         ;   vsetc r11                               }    
    {   ldw tail_mask, arg_job[JOB_LENGTH]      ;   mov zero_pv, arg_zero_pv                } // tail_mask holds length
    {   add Y, Y, tmp                           ;                                           }
    {   zext tail_mask, 5                       ;   shr length, tail_mask, 5                } // length holds length (in "vectors")
    {   shr tail_mask, tail_mask, 3             ;   mkmsk tmp, 4                            }
    {   mkmsk tail_mask, tail_mask              ;   bf length, .L_loop_end                  } // tail_mask is a byte mask

.L_loop_top:
    {   sub length, length, 1                   ;   vldr zero_pv[0]                         }
    OPTIONAL_DDR_SPILL(X)
    {   add X, X, _32                           ;   vlsub X[0]                              }
    {                                           ;   vdepth1                                 }
    {   vstrpv Y[0], tmp                        }
    {   add Y, Y, 4                             ;   bt length, .L_loop_top                  }
.L_loop_end:
    {                                           ;   bf tail_mask, .Lfunc_end                }
    {                                           ;   vldr zero_pv[0]                         }
    OPTIONAL_DDR_SPILL(X)
    {                                           ;   vlsub X[0]                              }
    {                                           ;   vdepth1                                 }
    {                                           ;   stw length, Y[0]                        } // Store 0 to upper bytes of tail
    {   vstrpv Y[0], tail_mask                  }

.Lfunc_end:

#ifdef USE_DDR_FIX
    {ldw ddr_spill, sp[2]                       ;   nop                                    }
#endif
    ldd r4, r5, sp[0]
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

