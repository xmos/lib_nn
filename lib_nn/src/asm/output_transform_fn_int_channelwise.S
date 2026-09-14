// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#define FUNCTION_NAME output_transform_fn_int_channelwise_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
	.align 4

VECT_INT16_MAX:
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80
	.word    0x7f807f80

VECT_INT8_MIN:
	.word    0x80808080
	.word    0x80808080
	.word    0x80808080
	.word    0x80808080

	.issue_mode  dual

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

    

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ; \
                              { ldw ddr_spill, x[0]; nop}
#warning ddr fix in use
#define ddr_spill            r5
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif

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

	.issue_mode  dual
	.globl FUNCTION_NAME
	.align 4
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    {dualentsp NSTACKWORDS; nop}

    //prologue
    std r4, r5, sp[0]

    //load 32 in r2    r11 contains pointer to A
    {ldc r2, 32                        ; add r11, r2, 0}
    //load r11[0] in vR add and r11 + 32 to vD
    {vldr r11[0]                       ; add r11, r2, r11}
    {vldd r11[0]                       ; shl r11, r2, 3}
    ldw r2, sp[S_OUTPUT_COUNT]
    {vsetc r11                         ; shl r11, r2, 1}

    OPTIONAL_DDR_SPILL(r3) // destroys r5
    {vlsat r3[0]                       ; ldc r4, PARAMS_final_shr}
    {ld16s r4, r0[r4]                  ; add r3, r3, r11 }

    OPTIONAL_DDR_SPILL(r3) // destroys r5
    { vlmul r3[0]                       ; add r3, r3, r11 }

    OPTIONAL_DDR_SPILL(r3) // destroys r5
    { vladd r3[0]                       ; ldaw r11, sp[S_TEMP_VECTOR] }
    { vstr r11[0]                       ; mkmsk r3, 2 }

    vlashr r11[0], r4

    {vstr r11[0]                        ; ldap r11, VECT_INT16_MAX }

    {vladd r11[0]                       ; nop }
    {vdepth1                            ; ldaw r11, sp[S_MASK]}

    vstrpv r11[0], r3

    ldap r11, VECT_INT8_MIN
    {vldr r11[0]                        ; ldaw r11, sp[S_TEMP_VECTOR] } 

    { ldw r4, sp[S_OUTPUT_COUNT]        ; nop }
    { mkmsk r3, r4                      ; ldw r2, sp[S_MASK] }
    vstrpv r1[0], r3                                          //store the int8_mins to output Y

    {vldr r11[0]                        ; add r0, r1, r4 }
    {vdepth8                            ; andnot r3, r2}
    vstrpv r1[0], r3

    //epilogue
    ldd r4, r5, sp[0]
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

#if (defined(__VX4A__) || defined(__VX4B__))

#define FUNCTION_NAME output_transform_fn_int_channelwise_asm

#define PARAMS_output_slice_channel_count 0
#define PARAMS_final_shr                  2       // int16

//Registers

//#define params_p             a0
//#define Y_p                  a1
//#define A_p                  a2
//#define otv_p                A_p
//#define out_ch_g             a3
//#define output_count         out_ch_g
//#define s                    a4

#define NSTACKWORDS     12

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    { li t3, 32 ; xm.entsp (NSTACKWORDS * 4) }

    { add a2, a2, t3 ; xm.vldr a2 }
    { xm.shl t3, t3, 3 ; xm.vldd a2 }
    mv a2, a4
    { xm.shl t3, a2, 1 ; xm.vsetc t3 }

    xm.vlsat a3
    add a3, a3, t3
    xm.vlmul0 a3
    xm.vlmul1 a3

    { add a3, a3, t3 ; xm.ldawsp t3, 0 * 4 }
    xm.vladd a3

    lh a4, PARAMS_final_shr * 2 (a0)
    { add a0, a1, a2 ; xm.vstr t3 }

    xm.vlashr t3, a4

    { xm.mkmsk a2, a2 ; xm.vdepth8 }
    xm.vstrpv a1, a2

    xm.retsp (NSTACKWORDS * 4)
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  (NSTACKWORDS * 4)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif // defined(__VX4A__) || defined(__VX4B__)
