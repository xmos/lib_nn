// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)

#define FUNCTION_NAME output_transform_fn_binary_impl_asm

    .text
	.p2align 2


// int8_t *output_transform_fn_binary_impl_asm(int8_t *Y,
//                                 VPURingBuffer *A, 
//                                 int32_t output_channel_group,
//                                 threshold_t *thresholds)

//Registers

#define Y_p                  x10
#define A_p                  x11
#define output_channel_group x12
#define thresholds_p         x13
#define t                    x28

#define NSTACKWORDS 0
	.globl FUNCTION_NAME
	.p2align 2
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    {x.entsp (NSTACKWORDS)*4;x.nop}

#define MODE_S16 0x100
    li t, MODE_S16

    {x.vsetc t                         ; x.shli t, output_channel_group, 5 }
    {add thresholds_p, thresholds_p, t ; addi t, A_p,0}
    {x.vldr t; x.nop}
    {x.vladd thresholds_p; x.ldawsp t, 0} //let's use sp[0] as a temp
    {x.vdepth1; x.ldcu x13, 0x3}
    x.vstrpv t, x13
    {x.ldcu x13, 0; x.ldwi t, 0(t)}
    x.st16 t,  x13(Y_p)
    {x.retsp (NSTACKWORDS)*4; addi Y_p, Y_p, 2}

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif


