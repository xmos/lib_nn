// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)

#define FUNCTION_NAME output_transform_fn_impl_asm

	.align 4

#define NSTACKBYTES 64
	.globl FUNCTION_NAME
	.p2align 2
	.type FUNCTION_NAME,@function

//int8_t *OT_int8::output_transform_fn( int32_t twice_output_slice_channel_count,  x10
//                                      int8_t *Y,                       x11
//                                      int16_t *multipliers_and_biases  x12
//                                      int32_t output_channel_group     x13)

//setc (256)  
//p = (twice_output_slice_channel_count - output_channel_group << 5) + multipliers_and_biases
//q = min(32, twice_output_slice_channel_count - output_channel_group << 5)
//vlsat4 p
//vlmul p
//vladd p+q
//vlshr p+q
//vdepth8
//vstrpv Y, mkmsk(q>>1)
//return Y+q

// x28 Temp
// assumes {vD:vR} contain the accumultor
FUNCTION_NAME:
    {x.ldcu x13, 32                     ; x.shli x28, x13, 5}
    { sub x10, x10, x28                 ; x.shli x28, x13, 4}
    {x.vsetc x28                        ; add x12, x12, x10  }
    min x28, x10, x13 
    x.vlsat4 x12
    { x.vlmul0 x12                      ; add x10, x28, x12 }
    x.vlmul1 x12  
    { x.vladd x10                       ; x.shri x12, x28, 1} 
    x.vlshl3  x10
    { x.vdepth8                         ; x.mkmsk x12, x12 }
    x.vstrpv x11, x12  
    {x.retsp 0                          ; add x10, x11, x28}

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif


