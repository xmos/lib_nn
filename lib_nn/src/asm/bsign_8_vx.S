// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__VX4A__)
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

#define NSTACKWORDS     (8)
    
.text
.globl FUNCTION_NAME
.p2align 4
.type FUNCTION_NAME,@function

#define arg_Y           x10
#define arg_X           x11
#define arg_zero_pv     x12
#define arg_job         x13

#define Y               arg_Y
#define X               arg_X
#define length          x12
#define tail_mask       x13
#define tmp             x18
#define _32             x19
#define zero_pv         x28

#define JOB_START       0
#define JOB_LENGTH      1

#define VSR_VALUE       XS1_VSR_TYPE_SET(0, XS1_VSETC_TYPE_INT8)

.skip 8
FUNCTION_NAME:
    x.entsp (NSTACKWORDS)*4
    x.stdsp  x19,x18,0

    {   x.ldwi tmp, (JOB_START)*4(arg_job)        ;   x.ldcu _32, 32                            }
    {   add X, X, tmp                             ;   x.nop                                     }
        x.ldcu x28, VSR_VALUE                                                        
    {   x.shri tmp, tmp, 3                        ;   x.vsetc x28}    
    {   x.ldwi tail_mask, (JOB_LENGTH)*4 (arg_job);   addi zero_pv, arg_zero_pv,0               } // tail_mask holds length
    {   add Y, Y, tmp                             ;   x.nop                                     }
    {   x.zexti tail_mask, 5                      ;   x.shri length, tail_mask, 5               } // length holds length (in "vectors")
    {   x.shri tail_mask, tail_mask, 3            ;   x.mkmski tmp, 4                           }
    {   x.mkmsk tail_mask, tail_mask              ;   x.brff length, .L_loop_end                } // tail_mask is a byte mask

.L_loop_top:
    {   x.vldr zero_pv                            ;   subi length, length, 1                    }
    {   add X, X, _32                             ;   x.vlsub X}
    {   x.nop                                     ;   x.vdepth1                                 }
        x.vstrpv Y, tmp                        
    {   addi Y, Y, 4                              ;   x.bt length, .L_loop_top                  }
.L_loop_end:
    {   x.nop                                     ;   x.brff tail_mask, .Lfunc_end              }
    {   x.vldr zero_pv                            ;   x.nop                                     }
    {   x.nop                                     ;   x.vlsub X}
    {   x.nop                                     ;   x.vdepth1                                 }
    {   x.nop                                     ;   x.stwi length, 0                        (Y)} // Store 0 to upper bytes of tail
        x.vstrpv Y, tail_mask                  

.Lfunc_end:
    x.lddsp  x19, x18,0
    x.retsp (NSTACKWORDS)*4
    
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif

