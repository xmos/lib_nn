// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#define FUNCTION_NAME output_transform_fn_binary_impl_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
	.align 4

	.issue_mode  dual

// int8_t *output_transform_fn_binary_impl_asm(int8_t *Y,
//                                 VPURingBuffer *A, 
//                                 int32_t output_channel_group,
//                                 threshold_t *thresholds)

//Registers

#define Y_p                  r0
#define A_p                  r1
#define output_channel_group r2
#define thresholds_p         r3
#define t                    r11

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ; \
                              { ldw ddr_spill, x[0]; nop}
#warning ddr fix in use
#define ddr_spill            r5
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif

#define NSTACKWORDS 0
	.issue_mode  dual
	.globl FUNCTION_NAME
	.align 4
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    {dualentsp NSTACKWORDS}

#define MODE_S16 0x100
    ldc t, MODE_S16
    {vsetc t; shl t, output_channel_group, 5 }
    {add thresholds_p, thresholds_p, t; mov t, A_p}
    {vldr t[0]; nop}  // Don't bother loading D
    {vladd thresholds_p[0]; ldaw t, sp[0]} //let's use sp[0] as a temp
    {vdepth1; ldc r3, 0x3}
    vstrpv t[0], r3
    {ldc r3, 0; ldw t, t[0]}
    st16 t, Y_p[r3]
    {retsp NSTACKWORDS; add Y_p, Y_p, 2}

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

#endif // __XS3A__

#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME output_transform_fn_binary_impl_asm
.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:
    xm.entsp 0
    xm.ldcu x10, 0
    xm.assert x10 //NOT IMPLEMENTED
    xm.retsp 0
.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  0
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
