// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * 
 *   typedef struct {
 *      int32_t col_stride;
 *      int32_t cols;
 *      int32_t row_stride;
 *      int32_t rows;
 *      int8_t scale[VPU_INT8_ACC_PERIOD];
 *   } avgpool_direct_valid_params;
 * 
 * C_API
 * void avgpool_direct_valid_xcore(
 *    vpu_ring_buffer_t* acc,
 *    const int8_t X[],
 *    const avgpool_direct_valid_params* params);
 */

// #if defined(__XS3A__)

#include "asm_constants.h"

/*

*/


#define FUNCTION_NAME avgpool_direct_valid_xcore

#define NSTACKVECTS   0
#define NSTACKWORDS   (2 + 8*(NSTACKVECTS))

#define PRM_COL_STRIDE  0
#define PRM_COLS        1
#define PRM_ROW_STRIDE  2
#define PRM_ROWS        3
#define PRM_SCALES      4

#define acc         r0
#define X           r1
#define params      r2
#define col_stride  r3

#define row_stride  r4
#define rows        r5
#define total_cols  params
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    std r4, r5, sp[0]
    ldc r11, VPU_MODE_8BIT
  { ldc r11, (PRM_SCALES * 4)             ; vsetc r11                             }

  { add r11, params, r11                  ; ldw col_stride, params[PRM_COL_STRIDE]}
  { ldc r11, 32                           ; vldc r11[0]                           }
  { add acc, acc, r11                     ; vclrdr                                }
  {                                       ; ldw row_stride, params[PRM_ROW_STRIDE]}

  {                                       ; ldw rows, params[PRM_ROWS]            }
  {                                       ; ldw total_cols, params[PRM_COLS]      }

  .L_loop_loop:
    { sub r11, r11, 1                       ; vlmacc X[0]                           }
    { add X, X, col_stride                  ; bt r11, .L_col_loop                   }
    { sub rows, rows, 1                     ; mov r11, total_cols                   }
    { add X, X, row_stride                  ; bt rows, .L_row_loop                  }
  .L_loop_bot:

  { ldc r11, 32                           ;                                       }
  { add acc, acc, r11                     ; vstr acc[0]                           }
  {                                       ; vstd acc[0]                           }

.L_cleanup:
    ldd r4, r5, sp[0]
    retsp NSTACKWORDS
.L_func_end:

.cc_bottom FUNCTION_NAME.function
.set FUNCTION_NAME.nstackwords,NSTACKWORDS; .globl FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;              .globl FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;             .globl FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;           .globl FUNCTION_NAME.maxchanends

.size FUNCTION_NAME, .L_func_end-FUNCTION_NAME

// #endif




