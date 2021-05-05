// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

//Registers

#define Y_p               r0
#define X_p               r1
#define K_p               r2
#define threshold_current r3 //threshold_current and D_p are never used at the same time
#define D_p               r3 

#define k_height_loop     r4
#define k_width_loop      r5
#define ic_loop_counter   r6
#define oc_loop_counter   r7
#define r_256_bits        r8
#define x_width_loop      r9
#define s                 r10
#define t                 r11


/*

int8_t *ImToColValid::memcopy_fn(int8_t *T, int8_t *X,
                                 int32_t output_v_coord, int32_t output_h_coord, int32_t output_c_coord)
*/

//#ifndef NN_USE_REF
//  #define FUNCTION_NAME im_to_col_valid_impl
//#else
//  #define FUNCTION_NAME im_to_col_valid_impl_asm
//#endif // NN_USE_REF

#define FUNCTION_NAME im_to_col_valid_impl_asm

#define NSTACKWORDS  40
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 2
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS


    
    retsp NSTACKWORDS

    .cc_bottom FUNCTION_NAME.function
    .set FUNCTION_NAME.nstackwords,NSTACKWORDS
    .globl FUNCTION_NAME.nstackwords
    .set FUNCTION_NAME.maxcores,1
    .globl FUNCTION_NAME.maxcores
    .set FUNCTION_NAME.maxtimers,0
    .globl FUNCTION_NAME.maxtimers
    .set FUNCTION_NAME.maxchanends,0
    .globl FUNCTION_NAME.maxchanends
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME
    .issue_mode  single

#endif



