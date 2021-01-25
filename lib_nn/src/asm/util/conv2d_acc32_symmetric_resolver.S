
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
    void conv2d_acc32_symmetric_resolver(
        nn_image_t* Y,
        const nn_conv2d_accumulator_params_t* acc_context,
        const nn_acc32_to_int8_params_t* op_params,
        const window_op_job_context_t* job_context,
        const nn_window_op_params_t* wop_params);
*/

#ifndef NN_USE_REF
  #define FUNCTION_NAME conv2d_acc32_symmetric_resolver
#else
  #define FUNCTION_NAME conv2d_acc32_symmetric_resolver_asm
#endif // NN_USE_REF

#define NSTACKVECTS  (1)
#define NSTACKWORDS  (4 + 8*NSTACKVECTS)
    
.text
.issue_mode  dual
.global FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

// Args on stack
#define STACK_WOP_PARAMS    (NSTACKWORDS + 1)

// Vectors on the stack
#define STACK_VEC_TMP       (NSTACKWORDS - 8*1)

// Args
#define y           r0
#define acc         r1
#define op_params   r2
#define job_ctx     r3

// Other
#define _32         r4

FUNCTION_NAME:
    {   dualentsp NSTACKWORDS                                                               }
    {   ldc _32, 32                             ;   stw r4, sp[1]                           }

// Determine the current channel output group and advance op_params to point at it.
// sizeof(nn_acc32_to_int8_params_t) = 32 * 5
    {    shl r3, r4, 5                          ;   ldw r11, job_ctx[JOB_CTX_CUR_OUT_CHAN]  }
    {    shr r11, r11, 5                        ;                                           }
    {    maccu r4, op_params, r3, r11                                                       }
#undef job_ctx      //r3
#define vec_tmp     r3
    {   ldaw vec_tmp, sp[STACK_VEC_TMP]         ;                                           }

// Load accumulators into vD:vR.
// Also set VPU mode to 16-bit
    {   add r11, acc, _32                       ;   vldd acc[0]                             }
    {   shl r11, _32, 3                         ;   vldr r11[0]                             }
    {                                           ;   vsetc r11                               }
#undef acc          //r1

// Apply the op_params values
    {   ldaw vec_tmp, sp[STACK_VEC_TMP]         ;   vlsat op_params[0]         /* shift1 */ }
    {   add op_params, op_params, _32           ;   vstr vec_tmp[0]                         }
    {   add op_params, op_params, _32           ;   vldc op_params[0]           /* scale */ }
    {                                           ;   vclrdr                                  }
    {   shl r11, _32, 4                         ;   vlmacc vec_tmp[0]                       }
    {   add op_params, op_params, _32           ;   vldc op_params[0]    /* offset_scale */ }
    {   add op_params, op_params, _32           ;   vlmacc op_params[0]        /* offset */ }
    {   mkmsk r11, 16                           ;   vsetc r11                               }
    {                                           ;   vlsat op_params[0]         /* shift2 */ }
    {   vstrpv y[0], r11                                                                    }


    {                                           ;   ldw r4, sp[1]                           }
    {                                           ;   retsp NSTACKWORDS                       }

.L_func_end:
.cc_bottom FUNCTION_NAME.function

.size FUNCTION_NAME, .L_func_end - FUNCTION_NAME

.set FUNCTION_NAME.nstackwords,NSTACKWORDS;         .global FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;                      .global FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;                     .global FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;                   .global FUNCTION_NAME.maxchanends




#endif



