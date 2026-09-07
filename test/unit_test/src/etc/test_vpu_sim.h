// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#ifndef TEST_VPU_SIM_H_
#define TEST_VPU_SIM_H_

#include <stdint.h>

#if defined(__VX4A__) || defined(__VX4B__)

static inline void vsetc(const unsigned mode) { asm volatile("li x28, %0\n xm.vsetc x28" :: "i"(mode) : "x28"); }
static inline void vclrdr(void) {asm volatile("xm.vclrdr");}
static inline void vldc(const void *ptr) {asm volatile("xm.vldc %0" :: "x"(ptr));}
static inline void vldd(const void *ptr) {asm volatile("xm.vldd %0" :: "x"(ptr));}
static inline void vldr(const void *ptr) { asm volatile("xm.vldr %0" :: "x"(ptr)); }
static inline void vlmaccr0(const void *ptr) { asm volatile("xm.vlmaccr0 %0" :: "x"(ptr)); }
static inline void vlmaccr1(const void *ptr) { asm volatile("xm.vlmaccr1 %0" :: "x"(ptr)); }
static inline void vlmaccrb(const void *ptr) { asm volatile("xm.vlmaccrb %0" :: "x"(ptr)); }
static inline void vlmacc0(const void *ptr) { asm volatile("xm.vlmacc0 %0" :: "x"(ptr)); }
static inline void vlmacc1(const void *ptr) { asm volatile("xm.vlmacc1 %0" :: "x"(ptr)); }
static inline void vladd(const void *ptr) { asm volatile("xm.vladd %0" :: "x"(ptr)); }
static inline void vlsub(const void *ptr) { asm volatile("xm.vlsub %0" :: "x"(ptr)); }
static inline void vlmul0(const void *ptr) { asm volatile("xm.vlmul0 %0" :: "x"(ptr)); }
static inline void vlmul1(const void *ptr) { asm volatile("xm.vlmul1 %0" :: "x"(ptr)); }
static inline void vlsat(const void *shifts) { asm volatile("xm.vlsat %0" :: "x"(shifts)); }
static inline void vlsat_fixed(const void *shifts) { asm volatile("xm.vlsat %0" :: "x"(shifts)); }
static inline void vlashr(const void *ptr, int8_t shift) { asm volatile("xm.vlashr %0, %1" :: "x"(ptr), "x"(shift)); }
static inline void vpos(void) { asm volatile("xm.vpos"); }
static inline void vstrpv(void *ptr, unsigned mask) { asm volatile("xm.vstrpv %0, %1" :: "x"(ptr), "x"(mask)); }
static inline void vstr(void *ptr) { asm volatile("xm.vstr %0" :: "x"(ptr)); }
static inline void vstd(void *ptr) { asm volatile("xm.vstd %0" :: "x"(ptr)); }
static inline void vstc(void *ptr) { asm volatile("addi x28, %0, 0\n xm.vstc x28" :: "x"(ptr) : "x28"); }
static inline void vdepth1(void) { asm volatile("xm.vdepth1"); }
static inline void vdepth8(void) { asm volatile("xm.vdepth8"); }
static inline void vdepth16(void) { asm volatile("xm.vdepth16"); }

// helpers
static inline void vlmul(const void *ptr) { vlmul0(ptr); vlmul1(ptr); }
static inline void vlmacc(const void *ptr) { vlmacc0(ptr); vlmacc1(ptr); }
static inline void vlmaccr(const void *ptr) { vlmaccr0(ptr); vlmaccr1(ptr); }
static inline void vlmaccr_binary(const void *ptr) { vlmaccrb(ptr); }

#elif defined(__XS3A__)

static inline void vsetc(const unsigned mode) { asm volatile("ldc r11, %0\n vsetc r11" :: "i"(mode) : "r11"); }
static inline void vclrdr(void) { asm volatile("vclrdr"); }
static inline void vsign(void) { asm volatile("vsign"); }
static inline void vlmaccr(const void *ptr) { asm volatile("vlmaccr %0[0]" :: "r"(ptr)); }
static inline void vlmaccr1(const void *ptr) { asm volatile("vlmaccr1 %0[0]" :: "r"(ptr)); }
static inline void vldd(const void *ptr) { asm volatile("vldd %0[0]" :: "r"(ptr)); }
static inline void vldr(const void *ptr) { register const int32_t *__ptr asm("r11") = (const int32_t *)ptr; asm volatile("vldr %0[0]" :: "r"(__ptr)); }
static inline void vlmacc(const void *ptr) { asm volatile("vlmacc %0[0]" :: "r"(ptr)); }
static inline void vldc(const void *ptr) { asm volatile("vldc %0[0]" :: "r"(ptr)); }
static inline void vladd(const void *ptr) { asm volatile("vladd %0[0]" :: "r"(ptr)); }
static inline void vlsub(const void *ptr) { asm volatile("vlsub %0[0]" :: "r"(ptr)); }
static inline void vlmul(const void *ptr) { asm volatile("vlmul %0[0]" :: "r"(ptr)); }
static inline void vlashr(const void *ptr, int8_t shift) { asm volatile("vlashr %0[0], %1" :: "r"(ptr), "r"(shift)); }
static inline void vstrpv(void *ptr, unsigned mask) { asm volatile("vstrpv %0[0], %1" :: "r"(ptr), "r"(mask)); }
static inline void vstr(void *ptr) { asm volatile("vstr %0[0]" :: "r"(ptr)); }
static inline void vstd(void *ptr) { asm volatile("vstd %0[0]" :: "r"(ptr)); }
static inline void vstc(void *ptr) { asm volatile("add r11, %0, 0" :: "r"(ptr) : "r11"); asm volatile("vstc r11[0]" ::: "r11"); }
static inline void vlsat(const void *shift) { asm volatile("vlsat %0[0]" :: "r"(shift)); }
static inline void vlsat_fixed(const void *shift) { asm volatile("vlsat %0[0]" :: "r"(shift)); }
static inline void vpos(void) { asm volatile("vpos"); }
static inline void vdepth8(void) { asm volatile("vdepth8"); }
static inline void vdepth1(void) { asm volatile("vdepth1"); }
static inline void vdepth16(void) { asm volatile("vdepth16"); }

// helpers
static inline void vlmaccr_binary(const void *ptr) { vlmaccr1(ptr); }

#else
#warning "Non xmos architecture"
#endif

#endif  // TEST_VPU_SIM_H_
