// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

.issue_mode  dual

#define PARAMS_bytes_per_h_line 0
#define PARAMS_bytes_per_pixel 1
#define PARAMS_input_height 2
#define PARAMS_input_width 3
#define PARAMS_input_channel_groups 4
#define PARAMS_T_rewind 5
#define PARAMS_horizontal_mem_stride 6
#define PARAMS_vertical_mem_stride 7
#define PARAMS_vstrpv_mask 8
#define PARAMS_dontzero 9

//Registers

#define params_p          r0
#define T_p               r1
#define output_v_coord    r3
#define T_rewind output_v_coord
#define S_output_h_coord  1
#define S_output_c_coord  2

#define ih                r4
#define iw                r5
#define ic                r6
#define iw_init           r7
#define ic_init           r8
#define thirty_two        r9
#define s                 r10
#define vertical_mem_stride s
#define t                 r11
#define X_p               r11
#define horizontal_mem_stride r0

#define FUNCTION_NAME im_to_col_valid_impl_asm

#define NSTACKWORDS  10
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

FUNCTION_NAME:
    dualentsp NSTACKWORDS

    std r4, r5, sp[0]
    std r6, r7, sp[1]
    std r8, r9, sp[2]
    std r10, T_p, sp[3]
    stw params_p, sp[8]

    {ldw s, params_p[PARAMS_bytes_per_pixel] ; nop }
    {ldw t, sp[NSTACKWORDS + S_output_h_coord] ; nop }
    mul t, t, s

    {ldw s, sp[NSTACKWORDS + S_output_c_coord]; add t, r2, t}
    {ldw s, params_p[PARAMS_bytes_per_h_line]; add t, t, s}
    mul s, output_v_coord, s
    {add r2, t, s ; ldw ih, params_p[PARAMS_input_height] }
    
    {ldw iw_init, params_p[PARAMS_input_width]; nop}
    {ldw ic_init, params_p[PARAMS_input_channel_groups] ; ldc thirty_two, 32 }
    {ldw T_rewind, params_p[PARAMS_T_rewind] ; nop }
    {ldw vertical_mem_stride, params_p[PARAMS_vertical_mem_stride] ; nop }
    {ldw r2, params_p[PARAMS_vstrpv_mask] ; sub X_p, r2, vertical_mem_stride }
    {ldw horizontal_mem_stride, params_p[PARAMS_horizontal_mem_stride] ; nop}

    ih_loop_asm:
        {mov iw, iw_init; add X_p, X_p, vertical_mem_stride}
        iw_loop_asm:
            {sub ic, ic_init, 1 ; bf  ic_init, skip_ic_loop_asm}
            
            ic_loop_asm:
                {vldd X_p[0]; add X_p, X_p, thirty_two}
                {vstd T_p[0]; add T_p, T_p, thirty_two}
            {bt ic, ic_loop_asm; sub ic, ic, 1}
        skip_ic_loop_asm:
            {vldr   X_p[0]; add X_p, X_p, thirty_two}
            vstrpv  T_p[0], r2

        {sub T_p, T_p, T_rewind; add X_p, X_p, horizontal_mem_stride}
        {bt iw, iw_loop_asm; sub iw, iw, 1}

    {bt ih, ih_loop_asm; sub ih, ih, 1}
    ldw params_p, sp[8]
    {ldw r2, params_p[PARAMS_dontzero] ; nop }
    {bt r2, dont_zero_asm                                ; nop }
    {vclrdr; nop}
    {vstd T_p[0];nop}
dont_zero_asm:  
    ldd r4, r5, sp[0]
    ldd r6, r7, sp[1]
    ldd r8, r9, sp[2]
    ldd r10, r0, sp[3]
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


#if defined(__VX4A__)


#define PARAMS_bytes_per_h_line 0
#define PARAMS_bytes_per_pixel 1
#define PARAMS_input_height 2
#define PARAMS_input_width 3
#define PARAMS_input_channel_groups 4
#define PARAMS_T_rewind 5
#define PARAMS_horizontal_mem_stride 6
#define PARAMS_vertical_mem_stride 7
#define PARAMS_vstrpv_mask 8
#define PARAMS_dontzero 9

//Registers

#define params_p          x10
#define T_p               x11
#define output_v_coord    x13
#define T_rewind output_v_coord
#define S_output_h_coord  x14
#define S_output_c_coord  x15

#define ih                x18
#define iw                x19
#define ic                x20
#define iw_init           x21
#define ic_init           x22
#define thirty_two        x23
#define s                 x24
#define vertical_mem_stride s
#define t                 x28
#define X_p               x28
#define horizontal_mem_stride x10

#define FUNCTION_NAME im_to_col_valid_impl_asm

#define NSTACKWORDS  16
    
.text
.globl FUNCTION_NAME
.p2align 2
.type FUNCTION_NAME,@function

FUNCTION_NAME:
    x.entsp (NSTACKWORDS)*4

    x.stdsp  x19, x18, 0
    x.stdsp  x21, x20, 8
    x.stdsp  x23, x22, 16
    x.stdsp  T_p, x24, 24
    x.stwsp params_p,  32

    {x.ldwi s, (PARAMS_bytes_per_pixel)*4 (params_p); x.nop }
    mul t, x14, s

    //{x.ldwsp s, (NSTACKWORDS + S_output_c_coord)*4; add t, x12, t}
    {x.nop; add t, x12, t}
    addi s, x15, 0
    {x.ldwi s, (PARAMS_bytes_per_h_line)*4(params_p); add t, t, s}
    mul s, output_v_coord, s
    {add x12, t, s ; x.ldwi ih, (PARAMS_input_height)*4 (params_p)}
    
    {x.ldwi iw_init, (PARAMS_input_width)*4(params_p); x.nop}
    {x.ldwi ic_init, (PARAMS_input_channel_groups)*4 (params_p); x.ldcu thirty_two, 32 }
    {x.ldwi T_rewind, (PARAMS_T_rewind)*4 (params_p); x.nop }
    {x.ldwi vertical_mem_stride, (PARAMS_vertical_mem_stride)*4 (params_p); x.nop }
    {x.ldwi x12, (PARAMS_vstrpv_mask)*4 (params_p); sub X_p, x12, vertical_mem_stride }
    {x.ldwi horizontal_mem_stride, (PARAMS_horizontal_mem_stride)*4 (params_p); x.nop}

    ih_loop_asm:
        {addi iw, iw_init,0; add X_p, X_p, vertical_mem_stride}
        iw_loop_asm:
            {subi ic, ic_init, 1 ; x.brff  ic_init, skip_ic_loop_asm}
            
            ic_loop_asm:
                {x.vldd X_p; add X_p, X_p, thirty_two}
                {x.vstd T_p; add T_p, T_p, thirty_two}
            {x.bt ic, ic_loop_asm; subi ic, ic, 1}
        skip_ic_loop_asm:
            {x.vldr   X_p; add X_p, X_p, thirty_two}
            x.vstrpv  T_p, x12

        {sub T_p, T_p, T_rewind; add X_p, X_p, horizontal_mem_stride}
        {x.bt iw, iw_loop_asm; subi iw, iw, 1}

    {x.bt ih, ih_loop_asm; subi ih, ih, 1}
    x.ldwsp params_p, 32
    {x.ldwi x12, (PARAMS_dontzero)*4 (params_p); x.nop }
    {x.bt x12, dont_zero_asm                                ; x.nop }
    {x.vclrdr; x.nop}
    {x.vstd T_p;x.nop}
dont_zero_asm:  
    x.lddsp  x19,x18,0
    x.lddsp  x21,x20,8
    x.lddsp  x23,x22,16
    x.lddsp  x10,x24,24
    x.retsp (NSTACKWORDS)*4
    
.Ltmp0:
    .size FUNCTION_NAME, .Ltmp0-FUNCTION_NAME

#endif




