// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * 
 *   typedef struct {
 *      int32_t col_stride;
 *      int32_t cols;
 *      int32_t row_stride;
 *      int32_t rows;
 *   } maxpool_direct_valid_params;
 * 
 * C_API
 * void maxpool_direct_valid_xcore(
 *    vpu_ring_buffer_t* acc,
 *    const int8_t* X,
 *    const maxpool_direct_valid_params* params);
 */

#if defined(__XS3A__)

#include "asm_constants.h"

/*

*/


#define FUNCTION_NAME maxpool_direct_valid_xcore

#define NSTACKVECTS   0
#define NSTACKWORDS   (4 + 8*(NSTACKVECTS))

#define PRM_COL_STRIDE  0
#define PRM_COLS        1
#define PRM_ROW_STRIDE  2
#define PRM_ROWS        3

#define acc         r0
#define arg_x       r1
#define params      r2
#define col_stride  r3

#define row_stride  r4
#define rows        r5
#define cols        r6

#define tmp         arg_x
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    std r6, r7, sp[1]
    ldc r11, VPU_MODE_8BIT

  { mov r11, arg_x                        ; vsetc r11                             }
  {                                       ; ldw col_stride, params[PRM_COL_STRIDE]}
  {                                       ; ldw cols, params[PRM_COLS]            }
  { sub cols, cols, 1                     ; ldw row_stride, params[PRM_ROW_STRIDE]}

  {                                       ; ldw rows, params[PRM_ROWS]            }
  { add r11, r11, col_stride              ; vldr r11[0]                           }
  {                                       ; vstr acc[0]                           }
    nop // guarantees 16-byte alignment of .L_col_loop, which means no FNOPs in the
        // inner loop.

  .L_row_loop:
    .L_col_loop:
      { sub cols, cols, 1                     ; vldr r11[0]                           }
      { ldc tmp, 32                           ; vlsub acc[0]                          }
      { add tmp, acc, tmp                     ; vdepth1                               }
      {                                       ; vstr tmp[0]                           }
      {                                       ; ldw tmp, tmp[0]                       }
      { add r11, r11, col_stride              ; vldr r11[0]                           }
        vstrpv acc[0], tmp
      {                                       ; bt cols, .L_col_loop                  }

    { sub rows, rows, 1                     ; ldw cols, params[PRM_COLS]            }
    { add r11, r11, row_stride              ; bt rows, .L_row_loop                  }
  .L_loop_bot:

.L_cleanup:
    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    retsp NSTACKWORDS
.L_func_end:

.cc_bottom FUNCTION_NAME.function
.set FUNCTION_NAME.nstackwords,NSTACKWORDS; .globl FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;              .globl FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;             .globl FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;           .globl FUNCTION_NAME.maxchanends

.size FUNCTION_NAME, .L_func_end-FUNCTION_NAME

#endif




