// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/*
  typedef struct {
    int32_t pixels;
    int8_t scale[VPU_INT8_ACC_PERIOD];
  } avgpool_patch_params;

  C_API
  void avgpool_patch_xcore(
       vpu_ring_buffer_t* acc,
       const int8_t patch[],
       const avgpool_patch_params* params)
 */

// #if defined(__XS3A__)

#include "asm_constants.h"



#define FUNCTION_NAME avgpool_patch_xcore

#define PRM_PIXELS  0
#define PRM_SCALES  1

#define NSTACKVECTS   0
#define NSTACKWORDS   (0 + 8*(NSTACKVECTS))

#define acc         r0
#define patch       r1
#define params      r2
#define _32         r3

#define pixels      params
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    ldc r11, VPU_MODE_8BIT
  { add r11, params, 4                    ; vsetc r11                             }
  { ldc _32, 32                           ; vclrdr                                }
  {                                       ; vldc r11[0]                           }
  {                                       ; ldw pixels, params[0]                 }

  // astew: This can be optimized further by unrolling the loop below. To maximize efficiency
  //        it might have to be 7 VLMACCs and a loop branch. That way it can all fit in the 
  //        instruction buffer at once and FNOPs can be eliminated. You'd wind up with 7 VLMACCs
  //        per 8 cycles. (instead of the 1:2 that this gets). Then the tail (mod 7) can be done
  //        less efficiently afterwards.
  //        But that would involve division by 7, which is best not done inside this function, so
  //        that would need to go into params.
  //        Unrolling the loop to 4 VLMACCs with a branch should give a 4:5 efficiency, which is
  //        reasonably close to 7:8.
  //        Unrolling to 8 VLMACCs with a branch means it won't fit in the instruction buffer and
  //        then there will need to be at least 2 FNOPs in the loop, giving an efficiency of
  //        8:11, so that's worse.
  .L_loop_top:
    { sub pixels, pixels, 1                 ; vlmacc patch[0]                       }
    { add patch, patch, _32                 ; bt pixels, .L_loop_top                }
  .L_loop_bot:

  { add acc, acc, _32                     ; vstr acc[0]                           }
  {                                       ; vstd acc[0]                           }

.L_cleanup:
    retsp NSTACKWORDS
.L_func_end:

.cc_bottom FUNCTION_NAME.function
.set FUNCTION_NAME.nstackwords,NSTACKWORDS; .globl FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;              .globl FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;             .globl FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;           .globl FUNCTION_NAME.maxchanends

.size FUNCTION_NAME, .L_func_end-FUNCTION_NAME

// #endif




