// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * 
 * C_API
 * void shift_int8_output_transform_asm(
 *    int8_t* output,
 *    const vpu_ring_buffer_t* acc,
 *    const int16_t* right_shifts,
 *    const int channel_count);
 * 
 */

#if defined(__XS3A__)

#include "asm_constants.h"


#define FUNCTION_NAME shift_int8_output_transform_asm

#define NSTACKVECTS   1
#define NSTACKWORDS   (4 + 8*(NSTACKVECTS))

#define STACK_VECT_TMP  (NSTACKWORDS - 8)

#define STACK_TMP       2

#define output  r0
#define acc     r1
#define shr     r2
#define chans   r3
#define tmp     r4
#define neg8    r5

#define _32     acc
#define mask    chans
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]

    ldc r11, VPU_MODE_16BIT
  { ldap r11, vpu_vect_0x80               ; vsetc r11                             }
  { mkmsk mask, chans                     ; vldr r11[0]                           }
    vstrpv output[0], mask
  { mov r11, acc                          ; ldc _32, 32                           }
  { add r11, r11, _32                     ; vldr r11[0]                           }
  { ldc neg8, 8                           ; vldd r11[0]                           }
  { ldaw r11, sp[STACK_VECT_TMP]          ; vlsat shr[0]                          }
  { neg neg8, neg8                        ; vstr r11[0]                           }
    ldap r11, vpu_vect_0x007F
  { mkmsk tmp, 4                          ; vladd r11[0]                          }
  { ldaw r11, sp[STACK_VECT_TMP]          ; vdepth1                               }
    vstrpv r11[0], tmp
  {                                       ; ldw tmp, sp[STACK_TMP]                }
  { andnot mask, tmp                      ;                                       }
    vlashr r11[0], neg8  
  {                                       ; vdepth8                               }
    vstrpv output[0], mask

    ldd r4, r5, sp[0]
    retsp NSTACKWORDS
.L_func_end:

.cc_bottom FUNCTION_NAME.function
.set FUNCTION_NAME.nstackwords,NSTACKWORDS; .globl FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1; .globl FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0; .globl FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0; .globl FUNCTION_NAME.maxchanends

.size FUNCTION_NAME, .L_func_end-FUNCTION_NAME

#endif


/*

The reason for the somewhat convoluted logic above is to avoid the symmetric saturation logic that the VPU uses when
applying a VLSAT, VLASHR or VDEPTH8. With 8-bit values, symmetric saturation means that values that should output
as -128 would actually be output as -127. So this function works around that

The logic of the above function is as follows:

- Set VPU to 16-bit mode
- Create a mask for our output channels (channel_mask)
- Fill the outputs with 0x80 (-128)
- Load the accumulators into vD:vR
- Apply the right-shifts (function arg) to get the 8-bit outputs  (VLSAT)
  - NOTE: but we're in 16-bit mode, so we have 8 extra sign bits for each channel
- Copy these 16-bit values to the stack
- Add 0x007F (127) to vR[], causing all values other than -128 to become non-negative (VLADD)
- Collapse the 16 elements into a single 16-bit mask (value_mask) at the bottom of vR[]  (VDEPTH1)
  - here negative values become a 1 bit and non-negatives become 0
  - (that means only -128's are 1)
- Load that mask into a register (via the stack)
- Our new mask becomes:  output_mask = (channel_mask & ~value_mask)
- Load the temporary values we saved to the stack earlier back into vR[], left-shifting 8 bits (VLASHR)
  - Loading them with the shift allows us to avoid having to switch the VPU to 8-bit mode.
- Collapse the 16-bit values in vR[] down to 8 bits
  - Note that symmetric saturation logic is applied here, meaning values that should be -0x80 are -0x7F instead
- Store the vR[k] to the output using output_mask
  - output_mask specifically avoids writing the values that *should be* -0x80 to the output, because we already
    wrote those to the output at the top.
    
*/



