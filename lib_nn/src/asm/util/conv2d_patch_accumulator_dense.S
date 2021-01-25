
#if defined(__XS3A__)



#define JOB_CTX_ROW             0
#define JOB_CTX_COL             1
#define JOB_CTX_CUR_OUT_CHAN    2
#define JOB_CTX_OUT_CHANS       3
#define JOB_CTX_FLAGS           4

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

#define ACC_CTX_BIASES      0
#define ACC_CTX_WEIGHTS     1


/*
    void conv2d_patch_accumulator_dense(
        const nn_image_t* p_X,
        const nn_conv2d_accumulator_params_t* acc_context,
        nn_acc32_vector_t* acc,
        const window_op_job_context_t* job_context,
        const nn_window_op_params_t* wop_params);
*/

#ifndef NN_USE_REF
  #define FUNCTION_NAME conv2d_patch_accumulator_dense
#else
  #define FUNCTION_NAME conv2d_patch_accumulator_dense_asm
#endif // NN_USE_REF

#define NSTACKVECTS  (0)
#define NSTACKWORDS  (8 + 8*NSTACKVECTS)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

// Stack args
#define STACK_WOP_PARAMS    (NSTACKWORDS + 1)


// Args
#define x                   r0
#define acc_ctx             r1
#define acc                 r2
#define job_ctx             r3

// Other
#define _32                 r4
#define win_bytes           r5
#define acc_offset          r6
#define macc_groups         r7
#define out_chans           r8
#define cur_cog             r9
#define tmp                 r10

FUNCTION_NAME:
    {   dualentsp NSTACKWORDS                                                               }
    {   std r4, r5, sp[0]                                                                   }
    {   std r6, r7, sp[1]                                                                   }
    {   std r8, r9, sp[2]                                                                   }
    {   ldc _32, 32                             ;   stw r10, sp[6]                          }
    {   shl r11, _32, 4                         ;                                           }
    {                                           ;   vsetc r11                               }
                                               

//  acc_offset = 2*(VPU_INT8_ACC_PERIOD - job_context->out_chans)
    {                                           ;   ldw cur_cog, job_ctx[JOB_CTX_CUR_OUT_CHAN]      }
    {   ldc macc_groups, 16                     ;   ldw out_chans, job_ctx[JOB_CTX_OUT_CHANS]       }
    {   sub acc_offset, macc_groups, out_chans  ;   ldw r11, sp[STACK_WOP_PARAMS]                   }
    {   shl acc_offset, acc_offset, 1           ;   shr cur_cog, cur_cog, 4                         }

//  win_bytes = wop_params->input.channels * wop_params->window.shape.height * wop_params->window.shape.width

    {   ldc macc_groups, 31                     ;   ldw win_bytes, r11[WOP_PRM_IN_CHANS]    }
    {                                           ;   ldw tmp, r11[WOP_PRM_WIN_SHAPE_HEIGHT]  }
    {   mul win_bytes, win_bytes, tmp                                                       }
    {                                           ;   ldw tmp, r11[WOP_PRM_WIN_SHAPE_WIDTH]   }
    {   mul win_bytes, win_bytes, tmp                                                       }
    
//  macc_groups = (wop_params->input.channels + (VPU_INT8_EPV - 1)) >> VPU_INT8_EPV_LOG2
    {   add macc_groups, macc_groups, win_bytes ;                                           }
    {   shr macc_groups, macc_groups, 5         ;                                           }

// Now go grab the correct set of biases, and copy them to the accumulators
//  bias_hi = acc_ctx.bias + (cur_cog << 6)
    {   shl tmp, cur_cog, 6                     ;   ldw r11, acc_ctx[ACC_CTX_BIASES]        }
    {   add r11, r11, tmp                       ;                                           }
    {   add r11, r11, _32                       ;   vldd r11[0]                             }
    {                                           ;   vldr r11[0]                             }
    {   add r11, acc, _32                       ;   vstd acc[0]                             }
    {                                           ;   vstr r11[0]                             }

// Next, we need to move to the current COG slice of K, then to the final channel of that.
// K = &acc_ctx->K[win_bytes * VPU_INT8_ACC_PERIOD * cur_cog]
// K = &K[(job_context->out_chans-1) * win_bytes];

    {                                           ;   ldw r1, acc_ctx[ACC_CTX_WEIGHTS]        }
#undef acc_ctx      //r1
#define k           r1

    {   shl r11, win_bytes, 4                   ;   sub k, k, win_bytes                     }
    {   maccu tmp, k, r11, cur_cog                                                          }
    {   maccu tmp, k, out_chans, win_bytes                                                  }


// Finally, we need the stride to move from the end of one CIG to the start of the next
// cig_stride = win_bytes * out_chans + VPU_INT8_EPV

// Aaand..
#define cig_stride      r8
    {   mul r8, out_chans, win_bytes                                                        }
    {   add r8, r8, _32                         ;                                           }
#undef out_chans    // r8

// Now do the work
// (Note: If there's a full 16 channels, we can subtract 5 instructions from this loop (and many
//        of the instructions above as well)

    {   sub r11, acc, acc_offset                ;   bu .L_loop_top                          }
    .align 16
    .L_loop_top:
        {   add r11, r11, _32                       ;   vldd r11[0]                             }
        {   sub macc_groups, macc_groups, 1         ;   vldr r11[0]                             }
        {   shr r11, acc_offset, 1                  ;   vldc x[0]                               }
        {   add x, x, _32                           ;   bru r11                                 }

        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }

        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }

        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }

        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }
        {   sub k, k, win_bytes                     ;   vlmaccr k[0]                            }

        {   add r11, acc, _32                       ;   vstd acc[0]                             }
        {   add k, k, cig_stride                    ;   vstr r11[0]                             }
        {   sub r11, acc, acc_offset                ;   bt macc_groups, .L_loop_top             }
    


    {                                           ;   ldw r10, sp[6]                          }
    {   ldd r8, r9, sp[2]                                                                   }
    {   ldd r6, r7, sp[1]                                                                   }
    {   ldd r4, r5, sp[0]                                                                   }
    {                                           ;   retsp NSTACKWORDS                       }

.L_func_end1:
.cc_bottom FUNCTION_NAME.function

.size FUNCTION_NAME, .L_func_end1 - FUNCTION_NAME

.set FUNCTION_NAME.nstackwords,NSTACKWORDS;         .global FUNCTION_NAME.nstackwords
.set FUNCTION_NAME.maxcores,1;                      .global FUNCTION_NAME.maxcores
.set FUNCTION_NAME.maxtimers,0;                     .global FUNCTION_NAME.maxtimers
.set FUNCTION_NAME.maxchanends,0;                   .global FUNCTION_NAME.maxchanends


#endif



