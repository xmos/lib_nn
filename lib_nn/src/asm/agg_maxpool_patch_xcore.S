// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * C_API
 * void maxpool_patch_xcore(
 *    vpu_ring_buffer_t* acc,
 *    const int8_t* patch,
 *    const int pixels);
 */

#if defined(__XS3A__)

#include "asm_constants.h"

/*
  Function logic is:
    - Using..
      - the output accumulator's vR[] to store hte current max (curmax[])
      - the output accumulator's vD[] as a temporary vector

    - Initialize curmax[] with the contents of the first pixel
    - Iterate over the remaining pixels in the patch:
      - Load the current pixel into vR[]
      - vR[k] = curmax[k] - vR[k]   (VLSUB)
        - (the channels where the new pixel value is larger than the max will be negative)
      - Create a bitmask where negative channels = 1 and non-negative channels = 0 (VDEPTH1)
        - (i.e.  bit[k] is 1 iff curmax[k] should be updated)
      - Load the current pixel into vR[] again
      - Use the mask to overwrite those curmax[k] values that are less than the current pixel
    - Done. curmax[] is already in accumulator.vR[], where it needs to be
*/


#define FUNCTION_NAME maxpool_patch_xcore

#define NSTACKVECTS   0
#define NSTACKWORDS   (0 + 8*(NSTACKVECTS))

#define acc         r0
#define arg_patch   r1
#define pix         r2
#define _32         r3

#define tmp         arg_patch
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    ldc r11, VPU_MODE_8BIT
  { mov r11, arg_patch                    ; vsetc r11                             }
  { ldc _32, 32                           ; vldr r11[0]                           }
  { sub pix, pix, 1                       ; vstr acc[0]                           }
  { add r11, r11, _32                     ; bf pix, .L_loop_bot                   }
  {                                       ; bu .L_loop_top                        }
  /////

  .align 16  // This prevents any need for FNOPs in the loop
  .L_loop_top:
    { sub pix, pix, 1                       ; vldr r11[0]                           }
    { add tmp, acc, _32                     ; vlsub acc[0]                          }
    {                                       ; vdepth1                               }
    {                                       ; vstr tmp[0]                           }
    {                                       ; ldw tmp, tmp[0]                       }
    {                                       ; vldr r11[0]                           }
      vstrpv acc[0], tmp
    { add r11, r11, _32                     ; bt pix, .L_loop_top                   }
  .L_loop_bot:

.L_cleanup:
    retsp NSTACKWORDS
.L_func_end:

.cc_bottom FUNCTION_NAME.function
.set FUNCTION_NAME.nstackwords,NSTACKWORDS; .globl FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;              .globl FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;             .globl FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;           .globl FUNCTION_NAME.maxchanends

.size FUNCTION_NAME, .L_func_end-FUNCTION_NAME

#endif




