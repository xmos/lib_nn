// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#if defined(__XS3A__)

/*
void lookup8_asm(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned elm_start,
    const unsigned elm_count);
*/

#define FUNCTION_NAME lookup8_asm

#define NSTACKWORDS  4
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 16
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define STACK_ELM_COUNT     (NSTACKWORDS+1)

#define arg_start r3

#define Y       r0
#define X       r1
#define LUT     r2
#define n       r3

#define data    r4
#define i       r5

#define BLOCK_COUNT_LOG2 (4)
#define BLOCK_SIZE  (3)
#define BLOCK   {                               ;   ld8u data, X[n]             };       \
                {   sub n, n, 1                 ;   ld8u data, LUT[data]        };       \
                    st8 data, Y[n];

FUNCTION_NAME:
    dualentsp NSTACKWORDS
        std r4, r5, sp[1]

    {   add Y, Y, arg_start             ;   add X, X, arg_start             }
    {                                   ;   ldw n, sp[STACK_ELM_COUNT]      }

    //  We'll deal with the tail first, where tail is the first (n % BLOCK_COUNT)
    //  elements. We deal with the tail be skipping (BLOCK_COUNT-(n % BLOCK_COUNT))
    //  blocks using a BRU instruction.

    //  i = BLOCK_SIZE*(BLOCK_COUNT-(n % BLOCK_COUNT))
    {   ldc i, (1<<BLOCK_COUNT_LOG2)    ;   mov r11, n                      }    
    {   zext r11, BLOCK_COUNT_LOG2      ;   sub X, X, 1                     }
    {   sub r11, i, r11                 ;   ldc i, BLOCK_SIZE               }
        mul i, r11, i

    {                                   ;   bf n, .Lfunc_end                }
    {                                   ;   bru i                           }

    .L_loop_top:
        // 16 blocks = 48 instructions. Unfortunately every cycle is a memory access, so there will also
        // be a lot of FNOPs
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
            BLOCK
        {                               ;   bt n, .L_loop_top           }
    



.Lfunc_end:
    ldd r4, r5, sp[1]
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

/*
void lookup8_asm(
    uint8_t* Y,
    const uint8_t* X,
    const uint8_t* lut,
    const unsigned elm_start,
    const unsigned elm_count);
*/

#define FUNCTION_NAME lookup8_asm

.p2align 2
.globl FUNCTION_NAME
.type FUNCTION_NAME,@function

#define STACK_S2            (0)
#define STACK_S3            (STACK_S2 + 4)
#define STACK_S4            (STACK_S3 + 4)
#define STACK_S5            (STACK_S4 + 4)
#define STACK_S6            (STACK_S5 + 4)
#define STACK_VEC1_TMP      (STACK_S6 + 4)
#define STACK_RA            (STACK_VEC1_TMP + 32)
#define NSTACKBYTES         (STACK_RA + 4)
#define ROUND_UP(X, Y)      (((X+Y-1)/Y)*Y)

#define out_data    a0
#define in_data     a1
#define lut         a2
#define start       a3
#define count       s2
#define mask        s3
#define counter     s4
#define thirty_two  s5
#define vec1_tmp    s6

FUNCTION_NAME:

    xm.entsp ROUND_UP(NSTACKBYTES, 16)
    xm.stdsp s2, s3, 0*8
    xm.stdsp s4, s5, 1*8
    sw s6, 2*8(sp)
    mv count, a4
    { add out_data, out_data, start         ; add in_data, in_data, start           }
    { addi vec1_tmp, sp, STACK_VEC1_TMP     ; xm.ldcu thirty_two, 32                }
    { xm.mkmski mask, 32                    ; sub out_data, out_data, thirty_two    }   // out_data substract 32 first for optimize in the loop
    
    // Set VPU to 8-bit mode
    li x28, 512
    { xm.vsetc x28                          ; xm.shri counter, count, 5 }
    xm.vldc lut

    FUNCTION_NAME.loop:
        { xm.vldr in_data                   ; add in_data, in_data, thirty_two }
        xm.vlookup lut, 0, 1
        xm.vlookup lut, 1, 2
        xm.vlookup lut, 2, 3
        xm.vlookup lut, 3, 4
        xm.vlookup lut, 4, 5
        xm.vlookup lut, 5, 6
        xm.vlookup lut, 6, 7
        xm.vlookup lut, 7, 0
        xm.bt counter, FUNCTION_NAME.save
        // In last loop - make mask for remaining elements
        xm.zexti count, 5
        xm.mkmsk mask, count

        FUNCTION_NAME.save:
            xm.vstd vec1_tmp
            { xm.vldr vec1_tmp                  ; add out_data, out_data, thirty_two    }
            xm.vstrpv out_data, mask
            { addi counter, counter, -1         ; xm.bt counter, FUNCTION_NAME.loop     }


    xm.lddsp s2, s3, 0*8
    xm.lddsp s4, s5, 1*8
    lw s6, 2*8(sp)

    xm.retsp ROUND_UP(NSTACKBYTES, 16)

.size FUNCTION_NAME, . -FUNCTION_NAME
.resource_const FUNCTION_NAME, "stack_frame_bytes",  ROUND_UP(NSTACKBYTES, 16)
.resource_list_empty FUNCTION_NAME, "callees"
.resource_list_empty FUNCTION_NAME, "tail_callees"
.resource_list_empty FUNCTION_NAME, "parallel_callees"

#endif
