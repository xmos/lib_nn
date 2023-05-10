// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)

#define FUNCTION_NAME output_transform_fn_int_channelwise_impl_asm

#define PARAMS_output_slice_channel_count 0
#define PARAMS_final_shr                  2       // int16

//Registers

//#define params_p             r0
//#define Y_p                  r1
//#define A_p                  r2
//#define otv_p                A_p     
//#define output_channel_group r3
//#define output_count output_channel_group
//#define s                    r4

#define t                    r11 

    //int8_t *OT_int8::output_transform_fn_int_channelwise( OT_int8_channelwise::Params *params,         r0
    //                                      int8_t *Y,                       r1
    //                                      VPURingBuffer *A                 r2
    //                                      int16_t *multipliers_and_biases  r3
    //                                      int32_t output_count             sp[OC])

    //  r2 holds 32.
    
#define S_R4            0
#define S_R5            1
#define S_MASK          2
#define S_TEMP_VECTOR   3
#define NSTACKWORDS     (S_TEMP_VECTOR + 1 + 8)         // Must be even
#define S_OUTPUT_COUNT  (NSTACKWORDS + 1)

	.globl FUNCTION_NAME
	.p2align 2
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    {x.entsp (NSTACKWORDS)*4; x.nop}

    //prologue
    x.stdsp  x19,x18,0

    //load 32 in r2    r11 contains pointer to A
    {x.ldcu x12, 32                        ; addi x28, x12, 0}
    //load r11[0] in vR add and r11 + 32 to vD
    {x.vldr x28; add x28, x12, x28}
    {x.vldd x28; x.shli x28, x12, 3}
    x.ldwsp x12, (S_OUTPUT_COUNT)*4
    {x.vsetc x28; x.shli x28, x12, 1}
//    {x.vlsat x13; x.ldcu x18, PARAMS_final_shr}
//{ld16s x18, x10, x18                  ; add x13, x13, x28 }

 //   { x.vlmul x13; add x13, x13, x28 }

    { x.vladd x13; x.ldawsp x28, (S_TEMP_VECTOR)*4 }
    { x.vstr x28; x.mkmski x13, 2 }

    x.vlashr x28, x18

   // {x.vstr x28; vs11 x28, VECT_INT16_MAX }

    {x.vladd x28; x.nop }
    {x.vdepth1                            ; x.ldawsp x28, (S_MASK)*4}

    x.vstrpv x28, x13

  //  lla x28, VECT_INT8_MIN
    {x.vldr x28; x.ldawsp x28, (S_TEMP_VECTOR)*4 } 

    { x.ldwsp x18, (S_OUTPUT_COUNT)*4        ; x.nop }
    { x.mkmsk x13, x18                      ; x.ldwsp x12, (S_MASK)*4 }
    x.vstrpv x11, x13                                          //store the int8_mins to output Y

    {x.vldr x28; add x10, x11, x18 }
//{vdepth8                            ; andnot x13, x12}"
    x.vstrpv x11, x13

    //epilogue
    x.lddsp  x19,x18,0
    x.retsp (NSTACKWORDS)*4
    
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif


