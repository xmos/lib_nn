

/*  
    void foo(void* p)
*/

#define FUNCTION_NAME   foo


#define NSTACKWORDS  (8)
    
.text
.issue_mode  dual
.globl FUNCTION_NAME
.align 4
.type FUNCTION_NAME,@function
.cc_top FUNCTION_NAME.function,FUNCTION_NAME

#define STACK_THREADSP

#define p       r0
#define sync    r1
#define tmp     r2
#define k       r3

FUNCTION_NAME:
    dualentsp NSTACKWORDS
    
    getr sync, 3
    ldc k, 5

    .L_init_loop:
        sub k, k, 1
        getst tmp, res[sync]
        ldap r11, .L_slave_start
        init t[tmp]:pc, r11
        set t[tmp]:r0, p
        set t[tmp]:r2, k
        bt k, .L_init_loop

    msync res[sync]
    mjoin res[sync]
    freer res[sync]
    bu .L_end

.L_slave_start:
    dualentsp 0
    shl r11, tmp, 6
    add r11, p, r11

    ldc k, 10000
    .L_thing:
        vldd r11[0]
        sub k, k, 1
        ldaw r11, r11[11]
        stw tmp, p[tmp]
        bt k, .L_thing

    ssync


.L_end:
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



