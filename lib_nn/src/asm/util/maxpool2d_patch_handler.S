
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


/*
    void maxpool2d_patch_handler(
        void* p_context,  
        nn_image_t* Y, 
        const nn_image_t* X, 
        const window_op_job_context_t* job_context,
        const nn_window_op_params_t* wop_params)
*/

#ifndef NN_USE_REF
  #define FUNCTION_NAME maxpool2d_patch_handler
#else
  #define FUNCTION_NAME maxpool2d_patch_handler_asm
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
#define context             r0
#define y                   r1
#define x                   r2
#define job_ctx             r3

// Other
#define _32                 r4
#define cur_chan            r5
#define x_chans             r6
#define pix_left            r7
#define tmp                 r10

FUNCTION_NAME:
    {   dualentsp NSTACKWORDS                                                               }
    {   std r4, r5, sp[0]                                                                   }
    {   std r6, r7, sp[1]                                                                   }
    {   std r8, r9, sp[2]                                                                   }
    {   ldc _32, 32                             ;   stw r10, sp[6]                          }

    // Set VPU mode to 8-bit
    {   shl r11, _32, 4                         ;                                           }
    {                                           ;   vsetc r11                               }

    // Initialize vec_curmax[] to -128
    {   ldaw r11, cp[vpu_vects_vec_0x80]                                                    }
    {                                           ;   vldd r11[0]                             }
    {                                           ;   vstd y[0]                               }

    {                                           ;   ldw cur_chan, job_ctx[JOB_CTX_CUR_OUT_CHAN]     }
#undef job_ctx  //r3
    {                                           ;   ldw r3, sp[STACK_WOP_PARAMS]                    }
    {                                           ;   ldw x_chans, r3[WOP_PRM_IN_CHANS]               }
    {   add x, x, cur_chan                      ;   ldw pix_left, r3[WOP_PRM_WIN_SHAPE_HEIGHT]      }
    {   mov r11, x                              ;   ldw r3, r3[WOP_PRM_WIN_SHAPE_WIDTH]             }
    {   mul pix_left, pix_left, r3                                                          }
#undef x    //r2    //X is just kept in r11 below


#define vec_curmax      r3

    .L_loop_top:
        {   ldaw r2, sp[STACK_WOP_PARAMS]           ;   vldr r11[0]                             }
        {   sub pix_left, pix_left, 1               ;   vlsub y[0]                              }
        {   mkmsk tmp, 4                            ;   vdepth1                                 }
        {   vstrpv r2[0], tmp                                                                   }
        {                                           ;   ldw tmp, r2[0]                          }
        {   add r11, r11, x_chans                   ;   vldr r11[0]                             }
        {   vstrpv y[0], tmp                                                                    }
        {                                           ;   bt pix_left, .L_loop_top                }
    .L_loop_bot:

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



