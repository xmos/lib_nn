// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)

#define FUNCTION_NAME output_transform_fn_impl_asm

	.align 4
NEGATIVE_VECT_SHIFT_TABLE:
	.word    0xFFF0FFF0, 0xFFF0FFF0, 0xFFF0FFF0, 0xFFF0FFF0, 0xFFF0FFF0, 0xFFF0FFF0, 0xFFF0FFF0, 0xFFF0FFF0
	.word    0xFFF1FFF1, 0xFFF1FFF1, 0xFFF1FFF1, 0xFFF1FFF1, 0xFFF1FFF1, 0xFFF1FFF1, 0xFFF1FFF1, 0xFFF1FFF1
	.word    0xFFF2FFF2, 0xFFF2FFF2, 0xFFF2FFF2, 0xFFF2FFF2, 0xFFF2FFF2, 0xFFF2FFF2, 0xFFF2FFF2, 0xFFF2FFF2
	.word    0xFFF3FFF3, 0xFFF3FFF3, 0xFFF3FFF3, 0xFFF3FFF3, 0xFFF3FFF3, 0xFFF3FFF3, 0xFFF3FFF3, 0xFFF3FFF3
	.word    0xFFF4FFF4, 0xFFF4FFF4, 0xFFF4FFF4, 0xFFF4FFF4, 0xFFF4FFF4, 0xFFF4FFF4, 0xFFF4FFF4, 0xFFF4FFF4
	.word    0xFFF5FFF5, 0xFFF5FFF5, 0xFFF5FFF5, 0xFFF5FFF5, 0xFFF5FFF5, 0xFFF5FFF5, 0xFFF5FFF5, 0xFFF5FFF5
	.word    0xFFF6FFF6, 0xFFF6FFF6, 0xFFF6FFF6, 0xFFF6FFF6, 0xFFF6FFF6, 0xFFF6FFF6, 0xFFF6FFF6, 0xFFF6FFF6
	.word    0xFFF7FFF7, 0xFFF7FFF7, 0xFFF7FFF7, 0xFFF7FFF7, 0xFFF7FFF7, 0xFFF7FFF7, 0xFFF7FFF7, 0xFFF7FFF7
	.word    0xFFF8FFF8, 0xFFF8FFF8, 0xFFF8FFF8, 0xFFF8FFF8, 0xFFF8FFF8, 0xFFF8FFF8, 0xFFF8FFF8, 0xFFF8FFF8
	.word    0xFFF9FFF9, 0xFFF9FFF9, 0xFFF9FFF9, 0xFFF9FFF9, 0xFFF9FFF9, 0xFFF9FFF9, 0xFFF9FFF9, 0xFFF9FFF9
	.word    0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA, 0xFFFAFFFA
	.word    0xFFFBFFFB, 0xFFFBFFFB, 0xFFFBFFFB, 0xFFFBFFFB, 0xFFFBFFFB, 0xFFFBFFFB, 0xFFFBFFFB, 0xFFFBFFFB
	.word    0xFFFCFFFC, 0xFFFCFFFC, 0xFFFCFFFC, 0xFFFCFFFC, 0xFFFCFFFC, 0xFFFCFFFC, 0xFFFCFFFC, 0xFFFCFFFC
	.word    0xFFFDFFFD, 0xFFFDFFFD, 0xFFFDFFFD, 0xFFFDFFFD, 0xFFFDFFFD, 0xFFFDFFFD, 0xFFFDFFFD, 0xFFFDFFFD
	.word    0xFFFEFFFE, 0xFFFEFFFE, 0xFFFEFFFE, 0xFFFEFFFE, 0xFFFEFFFE, 0xFFFEFFFE, 0xFFFEFFFE, 0xFFFEFFFE
	.word    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
VECT_SHIFT_TABLE:
	.word    0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000, 0x000000
	.word    0x010001, 0x010001, 0x010001, 0x010001, 0x010001, 0x010001, 0x010001, 0x010001
	.word    0x020002, 0x020002, 0x020002, 0x020002, 0x020002, 0x020002, 0x020002, 0x020002
	.word    0x030003, 0x030003, 0x030003, 0x030003, 0x030003, 0x030003, 0x030003, 0x030003
	.word    0x040004, 0x040004, 0x040004, 0x040004, 0x040004, 0x040004, 0x040004, 0x040004
	.word    0x050005, 0x050005, 0x050005, 0x050005, 0x050005, 0x050005, 0x050005, 0x050005
	.word    0x060006, 0x060006, 0x060006, 0x060006, 0x060006, 0x060006, 0x060006, 0x060006
	.word    0x070007, 0x070007, 0x070007, 0x070007, 0x070007, 0x070007, 0x070007, 0x070007
	.word    0x080008, 0x080008, 0x080008, 0x080008, 0x080008, 0x080008, 0x080008, 0x080008
	.word    0x090009, 0x090009, 0x090009, 0x090009, 0x090009, 0x090009, 0x090009, 0x090009
	.word    0x0A000A, 0x0A000A, 0x0A000A, 0x0A000A, 0x0A000A, 0x0A000A, 0x0A000A, 0x0A000A
	.word    0x0B000B, 0x0B000B, 0x0B000B, 0x0B000B, 0x0B000B, 0x0B000B, 0x0B000B, 0x0B000B
	.word    0x0C000C, 0x0C000C, 0x0C000C, 0x0C000C, 0x0C000C, 0x0C000C, 0x0C000C, 0x0C000C
	.word    0x0D000D, 0x0D000D, 0x0D000D, 0x0D000D, 0x0D000D, 0x0D000D, 0x0D000D, 0x0D000D
	.word    0x0E000E, 0x0E000E, 0x0E000E, 0x0E000E, 0x0E000E, 0x0E000E, 0x0E000E, 0x0E000E
	.word    0x0F000F, 0x0F000F, 0x0F000F, 0x0F000F, 0x0F000F, 0x0F000F, 0x0F000F, 0x0F000F
	.word    0x100010, 0x100010, 0x100010, 0x100010, 0x100010, 0x100010, 0x100010, 0x100010


#define NSTACKBYTES 64
	.globl FUNCTION_NAME
	.p2align 2
	.type FUNCTION_NAME,@function

#define PARAMS_output_slice_channel_count 0
#define PARAMS_initial_shift              2       // int16
#define PARAMS_final_shr                  3       // int16

//int8_t *output_transform_fn_impl_asm(
//        const OT_int8::Params *params,    x10
//        int8_t *Y,                        x11    
//        VPURingBuffer *A,                 x12
//        int32_t output_channel_group,     x13
//        int16_t *multipliers_and_biases)  x14
//
//  int ocg = output_channel_group * VPU_INT16_EPV * 2
//  int output_count = std::min( params->output_slice_channel_count*2 - ocg,
//         (int32_t)VPU_INT16_EPV*2);
//  multipliers_and_biases += ocg;

//    y = shr(shr_and_sat({vR, vD}, initial_shift) * M + A, final_shr)
//
//    *Y = y //for output_count bytes
//    initial_shift can be both positive and negative
//    final_shr can be both positive and negative
//

FUNCTION_NAME:
    {x.entsp NSTACKBYTES; x.ldcu x28, 32}

    lw x28, PARAMS_output_slice_channel_count(x10) 
    x.shli x28, x28, 1      //push this into the compiler
    x.shli x13, x13, 5
    add x14, x14, x13
    add x14, x14, x13
    {sub x13, x28, x13;  x.ldcu x28, 32}
    min x13, x28, x13 

    { x.vldr x12;     add x12, x12, x28}
    { x.vldd x12;     x.shli x28, x28, 3 }
    { x.vsetc x28;  x.nop}                      //set into 16 bit mode

    lw x10, 4(x10)  //{x.ldwi x10, 4(x10) ; x.nop} 
    srai x12, x10, 16
    {x.sexti x10, 16; x.nop}  //x10 is initial_shift, x12 is final_shr

    {x.ldap x28, VECT_SHIFT_TABLE; x.shli x10, x10, 5}
    {add x10, x10, x28; x.nop }
    x.vlsat4 x10

    mv x10, x14

    { x.vlmul0 x10                       ; add x28, x10, x13}
      x.vlmul1 x10   
    { x.vladd x28                        ; x.ldawsp x28, 4 } 
      x.vstr x28
      srli x13, x13, 1                    
      x.vlashr  x28, x12
    { x.vdepth8                          ; x.mkmsk x12, x13 }
      x.vstrpv x11, x12  
    {x.retsp NSTACKBYTES; add x10, x11, x13}

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif


