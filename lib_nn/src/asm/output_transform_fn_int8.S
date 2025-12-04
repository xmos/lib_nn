// Copyright 2020-2025 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

#define FUNCTION_NAME output_transform_fn_impl_asm

    .text
	.cc_top FUNCTION_NAME.function,FUNCTION_NAME
	.align 4
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

    

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ; \
                              { ldw ddr_spill, x[0]; nop}
#warning ddr fix in use
#define ddr_spill            r5
#else
#define OPTIONAL_DDR_SPILL(x) /* */
#endif

#define t                    r11 

    //int8_t *OT_int8::output_transform_fn( OT_int8::Params *params,         r0
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

    std r4, r5, sp[0]

    {ldc r2, 32                        ; add r11, r2, 0}
    {vldr r11[0]                       ; add r11, r2, r11}
    {vldd r11[0]                       ; shl r11, r2, 3}
    {vsetc r11                         ; ldc r4, PARAMS_initial_shift }

    {ld16s r4, r0[r4]                  ; ldc r5, 0 }
    {lss r5, r4, r5                    ; shl r2, r4, 5 }
    ldap r11, VECT_SHIFT_TABLE
    {bt r5, negshift_vlsat             ; add r5, r11, r2 }

    {vlsat r5[0]                       ; nop }
    ldw r2, sp[S_OUTPUT_COUNT]
    {bu vlsat_done                     ; shl r11, r2, 1 }
    
negshift_vlsat: 
    {vlsat r11[0]                      ; ldaw r11, sp[S_TEMP_VECTOR] }
    { vstr r11[0]                      ; nop }
    vlashr r11[0], r4
    ldw r2, sp[S_OUTPUT_COUNT]
    shl r11, r2, 1
    
vlsat_done: 

    OPTIONAL_DDR_SPILL(r3) // destroys r5
    { vlmul r3[0]                       ; add r3, r3, r11 }

    OPTIONAL_DDR_SPILL(r3) // destroys r5
    { vladd r3[0]                       ; ldaw r11, sp[S_TEMP_VECTOR] }
    { vstr r11[0]                       ; ldc r4, PARAMS_final_shr }
    
    { ld16s r4, r0[r4]                  ; nop }

    vlashr r11[0], r4

    {vstr r11[0]                        ; ldap r11, VECT_INT16_MAX }             //store off the unmodified vector

    {vladd r11[0]                       ; mkmsk r3, 2}
    {vdepth1                            ; ldaw r11, sp[S_MASK]}

    vstrpv r11[0], r3

    ldap r11, VECT_INT8_MIN
    {vldr r11[0]                        ; ldaw r11, sp[S_TEMP_VECTOR] } 

    { ldw r4, sp[S_OUTPUT_COUNT]       ; nop }
    { mkmsk r3, r4                      ; ldw r2, sp[S_MASK] }
    vstrpv r1[0], r3                                          //store the int8_mins to output Y

    {vldr r11[0]                        ; add r0, r1, r4 }
    {vdepth8                            ; andnot r3, r2}
    vstrpv r1[0], r3

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




#if defined(__VX4A__) || defined(__VX4B__)

#define FUNCTION_NAME output_transform_fn_impl_asm

#define NSTACKWORDS (8+2)
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

//int8_t *OT_int8::output_transform_fn( OT_int8::Params *params,         x10
//                                      int8_t *Y,                       x11
//                                      VPURingBuffer *A                 x12
//                                      int16_t *multipliers_and_biases  x13
//                                      int32_t output_count             x14)

#define S_TEMP_VECTOR 2

#define PARAMS_output_slice_channel_count 0
#define PARAMS_initial_shift              2       // int16
#define PARAMS_final_shr                  3       // int16

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.entsp ROUND_UP(NSTACKWORDS*4, 16); xm.nop}

    xm.stdsp x18, x19, 0*8

    {xm.ldcu x12, 32 ; xm.addi x28, x12, 0}
    {xm.vldr x28     ; xm.add x28, x12, x28}
    {xm.vldd x28     ; xm.shli x28, x12, 3}
    {xm.vsetc x28     ;  xm.ldcu x19, 0}
    xm.ldcu x18, PARAMS_initial_shift
    xm.ld16s x12, x18(x10) 

    //fill x28 with the initial_shift
    xm.ldawsp x28, 4*S_TEMP_VECTOR
    xm.st16 x12, x19(x28)
    addi x19, x19, 1
    xm.st16 x12, x19(x28)
    xm.ldwi x12, 0(x28)
    xm.stdi x12, x12, 8*0(x28)
    xm.stdi x12, x12, 8*1(x28)
    xm.stdi x12, x12, 8*2(x28)
    xm.stdi x12, x12, 8*3(x28)
    xm.vlsat x28

    slli x12, x14, 1
    xm.vlmul0 x13
    xm.vlmul1 x13                      
    {xm.nop ; xm.add x13, x13, x12 }
    { xm.vladd x13 ; xm.nop }

    xm.vstr x28

    li x12, PARAMS_final_shr
    xm.ld16s x12, x12(x10)
    xm.vlashr x28, x12
    xm.vdepth8
    addi x12, x14, 0
    xm.mkmsk x28, x12  
    xm.vstrpv x11, x28
    //slli x12, x12, 1
    xm.lddsp x18, x19, 0*8
    {xm.retsp ROUND_UP(NSTACKWORDS*4, 16); xm.add x10, x11, x12 }

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"


#endif


#if defined(poop)

    //int8_t *OT_int8::output_transform_fn( OT_int8::Params *params,         x10
    //                                      int8_t *Y,                       x11
    //                                      VPURingBuffer *A                 x12
    //                                      int16_t *multipliers_and_biases  x13
    //                                      int32_t output_count             x14)

#define FUNCTION_NAME output_transform_fn_impl_asm

#define NSTACKWORDS 0
#define ROUND_UP(X, Y) (((X+Y-1)/Y)*Y)

.p2align 1
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function
FUNCTION_NAME:

    {xm.ldcu x12, 32 ; xm.addi x28, x12, 0}
    {xm.vldr x28     ; xm.add x28, x12, x28}
    {xm.vldd x28     ; xm.shli x28, x12, 3}
    {xm.vsetc x28    ; xm.nop}
    mv x12, x14
    xm.vlsat4 x13
    { xm.vlmul0 x13  ; xm.add x10, x13, x12 }
    xm.vlmul1 x13  
    { xm.vladd x10   ; xm.mkmsk x28, x12  }
    xm.vlshl3 x10
    xm.vstrpv x11, x28
    ret

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes", ROUND_UP(NSTACKWORDS*4, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"


#endif