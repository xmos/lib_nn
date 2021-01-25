
#if defined(__XS3A__)

#define JOB_CTX_ROW             0
#define JOB_CTX_COL             1
#define JOB_CTX_CUR_OUT_CHAN    2
#define JOB_CTX_OUT_CHANS       2
#define JOB_CTX_FLAGS           3

#define WOP_PRM_IN_HEIGHT           0
#define WOP_PRM_IN_WIDTH            1
#define WOP_PRM_IN_CHANS            2
#define WOP_PRM_OUT_HEIGHT          3
#define WOP_PRM_OUT_WIDTH           4
#define WOP_PRM_OUT_CHANS           5
#define WOP_PRM_WIN_SHAPE_HEIGHT    6
#define WOP_PRM_WIN_SHAPE_WIDTH     7
#define WOP_PRM_WIN_START_ROW       8
#define WOP_PRM_WIN_START_COL       9
#define WOP_PRM_WIN_STRIDE_VERT     10
#define WOP_PRM_WIN_STRIDE_HORI     11
#define WOP_PRM_WIN_DIL_VERT        12
#define WOP_PRM_WIN_DIL_HORI        13

/*
    void conv2d_acc32_asymmetric_resolver(
        nn_image_t* Y,
        const nn_conv2d_accumulator_params_t* acc_context,
        const nn_acc32_to_int8_params_t* op_params,
        const window_op_job_context_t* job_context,
        const nn_window_op_params_t* wop_params);
*/

#ifndef NN_USE_REF
  #define FUNCTION_NAME conv2d_acc32_asymmetric_resolver
#else
  #define FUNCTION_NAME conv2d_acc32_asymmetric_resolver_asm
#endif // NN_USE_REF

#define NSTACKVECTS  (1)
#define NSTACKWORDS  (4 + 8*NSTACKVECTS)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

// Args on stack
#define STACK_WOP_PARAMS    (NSTACKWORDS + 1)

// Vectors on the stack
#define STACK_VEC_TMP       (NSTACKWORDS - 8*1)

// Other stack data
#define STACK_TMP           (2)

// Args
#define y           r0
#define acc         r1
#define op_params   r2
#define job_ctx     r3

// Other
#define _32         r4

FUNCTION_NAME:
    {   dualentsp NSTACKWORDS                                                               }
    {   ldc r4, 5                               ;   stw r4, sp[0]                           }

// Determine the current channel output group and advance op_params to point at it.
// sizeof(nn_acc32_to_int8_params_t* op_params) = 32 * 5
    {    shl r4, r4, 5                          ;   ldw r11, job_ctx[JOB_CTX_CUR_OUT_CHAN]  }
    {    shr r11, r11, 5                        ;   mkmsk r3, 16                            }
    {   maccu r4, op_params, r4, r11                                                        }
    {   ldc _32, 32                             ;                                           }
#undef job_ctx  //r3

// Load a vector full of -128 and store it in Y. Later on, we'll only store the values that shouldn't be -128
// Also set mode to 16-bit
    {   ldaw r11, cp[vpu_vects_vec_0x80]                                                    }
    {   shl r11, _32, 3                         ;   vldr r11[0]                             }
    {                                           ;   vsetc r11                               }
    {   vstrpv y[0], r3                                                                     }  
                                  

// Load the accumulator vectors into vD:vR
#define vec_tmp     r3
    {   add r11, acc, _32                       ;   vldd acc[0]                             }
    {   ldaw vec_tmp, sp[STACK_VEC_TMP]         ;   vldr r11[0]                             }
#undef acc          // r1
#define minus_8     r1

// Apply the shift and scale values, staying in 16-bit mode so that -128s don't get saturated to -127.
// Because the VPU is in 16-bit mode, each value occupies 2 bytes.
// 16-bit values get stored on the stack in vec_tmp.
    {   ldaw vec_tmp, sp[STACK_VEC_TMP]         ;   vlsat op_params[0]                      }
    {   add op_params, op_params, _32           ;   vstr vec_tmp[0]                         }
    {   add op_params, op_params, _32           ;   vldc op_params[0]                       }
    {   shr minus_8, _32, 2                     ;   vclrdr                                  }
    {   neg minus_8, minus_8                    ;   vlmacc vec_tmp[0]                       }
    {   add op_params, op_params, _32           ;   vldc op_params[0]                       }
    {   add op_params, op_params, _32           ;   vlmacc op_params[0]                     }
    {                                           ;   vlsat op_params[0]                      }
    {                                           ;   vstr vec_tmp[0]                         }
#undef op_params    //r2
#undef _32          //r4
#define tmpA        r2
#define tmpB        r4

// Add 127 to the 16-bit results now in vR. After adding, only values that were -128 will be negative.
// VDEPTH1 turns the elements into a 16-bit mask, indicating which elements would saturate in 8-bit mode.
    {   ldaw r11, cp[vpu_vects_vec_0x007F]                                                  }
    {   mkmsk r11, 4                            ;   vladd r11[0]                            }
    {   ldaw tmpA, sp[STACK_TMP]                ;   vdepth1                                 }

// Move the mask into a register
    {   vstrpv tmpA[0], r11                                                                 }
    {   mkmsk tmpB, 16                          ;   ldw tmpA, sp[STACK_TMP]                 }

// To make the elements 8-bits wide, we'll use VDEPTH8, which has an implicit 8-bit right-shift, so
// to compensate, we'll reload the 16-bit values from the stack using VLASHR with an 8-bit left-shift.
// (Also note that the mask we loaded needs to be inverted)
    {   vlashr vec_tmp[0], minus_8                                                          }
    {   andnot tmpB, tmpA                       ;   vdepth8                                 }

// Save the non-saturated results to Y
        vstrpv y[0], tmpB


    {                                           ;   ldw r4, sp[0]                           }
    {                                           ;   retsp NSTACKWORDS                       }

.L_func_end1:
.cc_bottom FUNCTION_NAME.function

.size FUNCTION_NAME, .L_func_end1 - FUNCTION_NAME

.set FUNCTION_NAME.nstackwords,NSTACKWORDS;         .global FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;                      .global FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;                     .global FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;                   .global FUNCTION_NAME.maxchanends



#endif



