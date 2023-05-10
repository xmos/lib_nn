// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__VX4A__)

#define FUNCTION_NAME output_transform_fn_int_clamped_impl_asm

#define PARAMS_output_slice_channel_count 0
#define PARAMS_initial_shift              2       // int16
#define PARAMS_final_shr                  3       // int16


//Registers

#define params_p             x10
#define output_slice_channel_count params_p
#define output_count output_slice_channel_count
#define Y_p                  x11
#define A_p                  x12  
#define output_channel_group x13

#define omb_p                x18 //offsets_multipliers_and_biases_p
#define initial_shift        x19   

#define final_shr            x20 
#define temp_p               x21 

#define t                    x28 

#ifdef USE_DDR_FIX
#define OPTIONAL_DDR_SPILL(x) { ldw ddr_spill, x[8]; nop} ; \
                              { ldw ddr_spill, x[0]; nop}
#warning ddr fix in use
#define ddr_spill              x19
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

#define NSTACKWORDS     (S_TEMP_VECTOR + 8 + 2)
#define S_OBM  (NSTACKWORDS + 1) //offsets_multipliers_and_biases_p

	.globl FUNCTION_NAME
	.p2align 2
	.type FUNCTION_NAME,@function

FUNCTION_NAME:
    x.entsp (NSTACKWORDS)*4

    x.stdsp  x19,x18,0
    x.stdsp  x21,x20,8

    {x.ldwi initial_shift, 4(params_p); x.shli output_channel_group, output_channel_group, 4}
    {x.ldwi output_slice_channel_count, 0(params_p); x.shri final_shr, initial_shift, 16}
    {sub output_count, output_slice_channel_count, output_channel_group; x.ldcu t, 16}
    {x.lss t, output_count, t; x.sexti final_shr, 16}

    {x.bt t, output_transform_fn_int_clamped_impl_asm_skip; x.sexti initial_shift, 16}

        {x.ldcu output_count, 16; x.ldcu t, 1}
    output_transform_fn_int_clamped_impl_asm_skip:
    
    {x.ldwsp omb_p, (S_OBM)*4; x.shli t, t, 8}

    //{mv t, A_p ; x.vsetc t}


   // lda16 temp_p, output_channel_group[output_channel_group ]
   // lda16 omb_p, omb_p[temp_p ]

    {x.vldr t; x.nop}


    {x.vladd omb_p; x.shli output_count, output_count, 1}
    {x.vpos; x.ldawsp temp_p, (S_TEMP_VECTOR)*4}
    
    {x.vstr temp_p; add omb_p, omb_p, output_count} 
    x.vlashr temp_p, initial_shift
    //{x.vlmul omb_p; add omb_p, omb_p, output_count}
    {x.vladd omb_p; x.shri output_count, output_count, 1}
    {x.vstr temp_p; x.nop}
    x.vlashr temp_p, final_shr

//{vstr temp_p[0 ]; *ldap t, VECT_INT16_MAX }

    {x.vladd t; x.mkmski x20, 2}
    {x.vdepth1                            ; x.ldawsp t, (S_MASK)*4}

    x.vstrpv t, x20

//ldap t, VECT_INT8_MIN
//{vldr t[0                        ]; ldawsp t, sp[S_TEMP_VECTOR ]}

    { x.mkmsk x13, x18                      ; x.ldwsp x12, (S_MASK)*4 }
    x.vstrpv x11, x13                                          //store the int8_mins to output Y

//{vldr t[0                        ]; add r0, Y_p, output_count }
//{vdepth8                            ; andnot x13, x12}
    x.vstrpv Y_p, x13

    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.retsp (NSTACKWORDS)*4

.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif


