// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.section        .cp.const32,"aMc",@progbits,32
.cc_top VECT_INT16_MAX.data
.align 4
VECT_INT16_MAX:
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.word    0x7f007f00
.cc_bottom VECT_INT16_MAX.data

.section        .cp.const32,"aMc",@progbits,32
.cc_top VECT_INT8_MIN.data
.align 4
VECT_INT8_MIN:
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.word    0x80808080
.cc_bottom VECT_INT8_MIN.data
.text

.issue_mode  dual

#define PARAMS_output_slice_channel_count 0
#define PARAMS_otv_p 1
#define PARAMS_biases 2
#define PARAMS_multipliers 3

//OutputTransformValues
#define OTV_bias_multipler 0
#define OTV_final_shr 8
#define OTV_accu_shr 16
#define OTV_accu_shl 24

//Registers

#define params_p             r0
#define Y_p                  r1
#define A_p                  r2
#define otv_p                A_p     
#define output_channel_group r3
#define output_count output_channel_group
#define s                    r4
#define t                    r11 

#define S_R4    0
#define S_MASK  1
#define S_TEMP_VECTOR 2

#define FUNCTION_NAME output_transform_fn_impl_asm

//int8_t *OT_int8::output_transform_fn(int8_t *Y, VPURingBuffer *A, int32_t output_channel_group)

#define NSTACKWORDS  63
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    stw r4, sp[0]

    {ldc A_p, 32; mov t, A_p}
    {vldr t[0]; add t, A_p, t}
    {vldd t[0]; shl t, A_p, 3}
    {vsetc t;  shl output_channel_group, output_channel_group, 5}

    {ldw otv_p, params_p[PARAMS_otv_p]; ldc t, OTV_accu_shr}
    ldaw t, otv_p[t]

    {vlsat t[0]; ldaw s, sp[S_TEMP_VECTOR]}
    {vstr s[0]; ldc t, OTV_accu_shl}
    {ldw t, otv_p[t]; nop}
    vlashr s[0], t
    {vstr s[0]; nop}

    {ldw t, params_p[PARAMS_biases]; nop}
    {vclrdr; add t, t, output_channel_group}
    {vldc t[0]; nop}
    ldaw t, otv_p[OTV_bias_multipler]
    {vlmacc t[0]; nop}

    {ldw t, params_p[PARAMS_multipliers]; nop}
    {vldc s[0]; add t, t, output_channel_group}
    {vlmacc t[0]; nop}

    ldaw t, otv_p[OTV_final_shr]
    {vlsat t[0];shr output_channel_group, output_channel_group, 1 }

    {ldw t, params_p[PARAMS_output_slice_channel_count]; ldaw s, sp[S_TEMP_VECTOR]}
    {sub t, t, output_channel_group; ldc output_channel_group, 16}
    {lss output_channel_group, t, output_channel_group; nop}
    {bt output_channel_group, asm_min_skip; mov output_count, t}
        {ldc output_count, 16; nop}
   
    asm_min_skip:

    {vstr s[0]; nop}             //store off the unmodified vector

    ldaw t, cp[VECT_INT16_MAX]
    {vladd t[0]; mkmsk t, 2}
    {vdepth1; ldaw s, sp[S_MASK]}

    vstrpv s[0], t 

    ldaw t, cp[VECT_INT8_MIN]

    {vldr t[0]; mkmsk t, output_count} 
    vstrpv Y_p[0], t    //store the int16_mins to output Y

    ldaw t, sp[S_TEMP_VECTOR]
    {vldr t[0]; mkmsk t, output_count}
    {ldw s, s[0]; nop}
    {vdepth8; andnot t, s}
    vstrpv Y_p[0], t

    {ldw r4, sp[0];add Y_p, Y_p, output_count}
    {retsp NSTACKWORDS; mov r0, Y_p}

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



