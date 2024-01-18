// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#define FUNCTION_NAME output_transform_maxpool_impl_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME

	.issue_mode  dual

#define PARAMS_output_slice_channel_count 0
#define PARAMS_initial_shift              2       // int16
#define PARAMS_final_shr                  3       // int16

//Registers

//#define params_p             r0
//#define Y_p                  r1
//#define A_p                  r2
//#define otv_p                A_p     
//#define output_channel_group r3
//#define output_count output_channel_group
//#define s                    r4

    

#define t                    r11 

    //int8_t *OT_int8::output_transform_maxpool( Ignored                     r0
    //                                      int8_t *Y,                       r1
    //                                      VPURingBuffer *A                 r2
    //                                      Ignored                          r3
    //                                      int32_t output_count             sp[OC])

    //  r2 holds 32.
    
#define NSTACKWORDS     (0)              
#define S_OUTPUT_COUNT  (NSTACKWORDS + 1)

	.issue_mode  dual
	.globl FUNCTION_NAME
	.align 4
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    { dualentsp NSTACKWORDS            ; add r11, r2, 0 }
    { nop                              ; ldw r3, sp[S_OUTPUT_COUNT] }
    { vldr r11[0]                      ; mkmsk r2, r3 }
    vstrpv r1[0], r2
    { retsp NSTACKWORDS                ; add r0, r1, r3 }

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


