// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#define FUNCTION_NAME output_transform_fn_int_clamped_impl_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
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

VECT_INT8_MIN:
	.word    0x80808080
	.word    0x80808080
	.word    0x80808080
	.word    0x80808080

	.issue_mode  dual

#define PARAMS_output_slice_channel_count 0
#define PARAMS_initial_shift              2       // int16
#define PARAMS_final_shr                  3       // int16


//Registers

#define params_p             r0
#define output_slice_channel_count params_p
#define output_count output_slice_channel_count
#define Y_p                  r1
#define A_p                  r2  
#define output_channel_group r3

#define omb_p                r4 //offsets_multipliers_and_biases_p
#define initial_shift        r5   

#define final_shr            r6 
#define temp_p               r7 

#define t                    r11 

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ; \
                              { ldw ddr_spill, x[0]; nop}
#warning ddr fix in use
#define ddr_spill              r5
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif


//int8_t *output_transform_fn_int_clamped_impl_asm(const OT_int8_clamped::Params *params, int8_t *Y,
//                                 VPURingBuffer *A, int32_t output_channel_group,
//                                 int16_t *offsets_multipliers_and_biases) {
    
#define S_R4            0
#define S_R5            1
#define S_R6            2
#define S_R7            3 
#define S_MASK          4 
#define S_unused        5 
#define S_TEMP_VECTOR   6

#define NSTACKWORDS     (S_TEMP_VECTOR + 8)
#define S_OBM  (NSTACKWORDS + 1) //offsets_multipliers_and_biases_p

	.issue_mode  dual
	.globl FUNCTION_NAME
	.align 4
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]

    {ldw initial_shift, params_p[1]; shl output_channel_group, output_channel_group, 4}
    {ldw output_slice_channel_count, params_p[0]; shr final_shr, initial_shift, 16}
    {sub output_count, output_slice_channel_count, output_channel_group; ldc t, 16}
    {lss t, output_count, t; sext final_shr, 16}
    {bt t, output_transform_fn_int_clamped_impl_asm_skip; sext initial_shift, 16}
        {ldc output_count, 16; ldc t, 1}
    output_transform_fn_int_clamped_impl_asm_skip:
    
    {ldw omb_p, sp[S_OBM]; shl t, t, 8}
    {vsetc t ; mov t, A_p }

    lda16 temp_p, output_channel_group[output_channel_group] //x3
    lda16 omb_p, omb_p[temp_p] //offset omb_p by output_channel_group * 16 * 3

    {vldr t[0]; nop}
    {vladd omb_p[0]; shl output_count, output_count, 1}
    {vpos; ldaw temp_p, sp[S_TEMP_VECTOR]}
    
    {vstr temp_p[0]; add omb_p, omb_p, output_count} 
    vlashr temp_p[0], initial_shift
    {vlmul omb_p[0]; add omb_p, omb_p, output_count}
    {vladd omb_p[0]; shr output_count, output_count, 1}
    {vstr temp_p[0]; nop}
    vlashr temp_p[0], final_shr

    {vstr temp_p[0] ; ldap t, VECT_INT16_MAX }             //store off the unmodified vector

    {vladd t[0]                         ; mkmsk r6, 2}
    {vdepth1                            ; ldaw t, sp[S_MASK]}

    vstrpv t[0], r6

    ldap t, VECT_INT8_MIN
    {vldr t[0]                        ; ldaw t, sp[S_TEMP_VECTOR] } 

    { mkmsk r3, r4                      ; ldw r2, sp[S_MASK] }
    vstrpv r1[0], r3                                          //store the int8_mins to output Y

    {vldr t[0]                        ; add r0, Y_p, output_count }
    {vdepth8                            ; andnot r3, r2}
    vstrpv Y_p[0], r3

    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
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


