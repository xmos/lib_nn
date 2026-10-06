// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

// Randomised differential tests for the VPU C model in vpu_sim.c.
//
// Each case loads identical vR/vD/vC state and memory operands into the hardware
// VPU and into the model, runs one operation (or a chain of them) on both, and
// requires the complete vR/vD/vC state afterwards to be bit-exact. Inputs mix
// uniform random lanes with small values and edge values (0, +-1, max, -max, min)
// so that rounding and saturation behaviour is exercised in every mode.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "unity_fixture.h"

#include "vpu_sim.h"
#include "xs3_vpu.h"
#include "nn_arch.h"

#include "tst_common.h"
#ifndef TEST_BUILD_NATIVE
#include "etc/test_vpu_sim.h"

#ifndef VPU_SIM_RANDOM_ITERS
#define VPU_SIM_RANDOM_ITERS 50
#endif
#define MAX_CHAIN 20     // longest chain of accumulating instructions
#define MAX_REPORTED 3   // mismatches printed in full per op and mode

typedef struct {
  vpu_vector_t vR, vD, vC;
} vpu_state_t;

#if defined(__xcore__) || defined(__riscv_xxcore)
#define HW_OP_FPTR __attribute__((fptrgroup("vpu_sim_random_hw")))
#define SIM_OP_FPTR __attribute__((fptrgroup("vpu_sim_random_sim")))
#else
#define HW_OP_FPTR
#define SIM_OP_FPTR
#endif

typedef void (*hw_op_t)(const vpu_vector_t *mem, int arg);
typedef void (*sim_op_t)(xs3_vpu *vpu, const vpu_vector_t *mem, int arg);

typedef enum {
  MEM_DATA,    // random lanes of the current mode
  MEM_SHIFTS,  // VLSAT shift vector
} mem_kind_t;

typedef struct {
  const char *name;
  HW_OP_FPTR hw_op_t hw;
  SIM_OP_FPTR sim_op_t sim;
  mem_kind_t mem_kind;
  int arg_min, arg_max;  // shift for VLASHR, chain length for the MACs
} vpu_case_t;

static const char *mode_name(unsigned mode) {
  switch (mode) {
    case MODE_S8: return "S8";
    case MODE_S16: return "S16";
    case MODE_S32: return "S32";
    default: return "?";
  }
}

static unsigned mode_bits(unsigned mode) {
  return mode == MODE_S8 ? 8 : mode == MODE_S16 ? 16 : 32;
}

static int rand_range(int lo, int hi) {
  return lo + (int)(pseudo_rand_uint32() % (uint32_t)(hi - lo + 1));
}

static int64_t edge_value(unsigned bits) {
  const int64_t max = ((int64_t)1 << (bits - 1)) - 1;
  switch (pseudo_rand_uint32() % 8) {
    case 0: return 0;
    case 1: return 1;
    case 2: return -1;
    case 3: return max;
    case 4: return -max;
    case 5: return -max - 1;
    case 6: return max / 2 + 1;
    default: return -(max / 2 + 1);
  }
}

// 32-bit arithmetic only: 64-bit random numbers make xsim runs several times slower
static int64_t rand_lane(unsigned bits) {
  const uint32_t r = pseudo_rand_uint32() % 8;
  if (r < 2) return edge_value(bits);
  int32_t v = (int32_t)pseudo_rand_uint32();
  if (r < 5) v >>= 32 - bits + rand_range(0, bits - 1);  // small magnitude
  return v;
}

static void fill_vector(vpu_vector_t *v, unsigned bits) {
  for (unsigned i = 0; i < 256 / bits; i++) {
    const int64_t x = rand_lane(bits);
    if (bits == 8) v->s8[i] = (int8_t)x;
    else if (bits == 16) v->s16[i] = (int16_t)x;
    else v->s32[i] = (int32_t)x;
  }
}

static void fill_shifts(vpu_vector_t *v, unsigned mode) {
  memset(v, 0, sizeof(*v));
  if (mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_ACC_PERIOD; i++) v->u32[i] = rand_range(0, 31);
  } else {
    for (int i = 0; i < VPU_INT16_ACC_PERIOD; i++) v->u16[i] = rand_range(0, 23);
  }
}

static void hw_run(unsigned mode, const vpu_state_t *in, const vpu_vector_t *mem,
                   HW_OP_FPTR hw_op_t op, int arg, vpu_state_t *out) {
  vsetc_reg(mode);
  vldr(&in->vR);
  vldd(&in->vD);
  vldc(&in->vC);
  op(mem, arg);
  vstr(&out->vR);
  vstd(&out->vD);
  vstc(&out->vC);
}

static void sim_run(unsigned mode, const vpu_state_t *in, const vpu_vector_t *mem,
                    SIM_OP_FPTR sim_op_t op, int arg, vpu_state_t *out) {
  xs3_vpu vpu;
  memset(&vpu, 0, sizeof(vpu));
  VSETC(&vpu, (vector_mode)mode);
  VLDR(&vpu, &in->vR);
  VLDD(&vpu, &in->vD);
  VLDC(&vpu, &in->vC);
  op(&vpu, mem, arg);
  VSTR(&vpu, &out->vR);
  VSTD(&vpu, &out->vD);
  VSTC(&vpu, &out->vC);
}

static void print_words(const char *label, const vpu_vector_t *v) {
  printf("    %-8s", label);
  for (int i = 0; i < VPU_INT32_EPV; i++) printf(" %08X", (unsigned)v->u32[i]);
  printf("\n");
}

static void report(const vpu_case_t *c, unsigned mode, int arg, const vpu_state_t *in,
                   const vpu_vector_t *mem, const vpu_state_t *hw, const vpu_state_t *sim) {
  printf("  MISMATCH %s %s arg=%d\n", c->name, mode_name(mode), arg);
  print_words("in.vR", &in->vR);
  print_words("in.vD", &in->vD);
  print_words("in.vC", &in->vC);
  print_words("mem[0]", &mem[0]);
  const char *names[3] = {"vR", "vD", "vC"};
  const vpu_vector_t *h[3] = {&hw->vR, &hw->vD, &hw->vC};
  const vpu_vector_t *s[3] = {&sim->vR, &sim->vD, &sim->vC};
  for (int r = 0; r < 3; r++) {
    if (memcmp(h[r], s[r], sizeof(vpu_vector_t)) == 0) continue;
    char label[16];
    snprintf(label, sizeof(label), "hw.%s", names[r]);
    print_words(label, h[r]);
    snprintf(label, sizeof(label), "sim.%s", names[r]);
    print_words(label, s[r]);
  }
}

// Runs one case in every listed mode and returns the number of mismatching iterations.
static unsigned run_case(const vpu_case_t *c, const unsigned *modes, unsigned n_modes) {
  unsigned total = 0;
  for (unsigned m = 0; m < n_modes; m++) {
    const unsigned mode = modes[m];
    const unsigned bits = mode_bits(mode);
    unsigned fails = 0;
    for (unsigned it = 0; it < VPU_SIM_RANDOM_ITERS; it++) {
      vpu_state_t WORD_ALIGNED in, hw, sim;
      vpu_vector_t WORD_ALIGNED mem[MAX_CHAIN];
      fill_vector(&in.vR, bits);
      fill_vector(&in.vD, bits);
      fill_vector(&in.vC, bits);
      const int arg = rand_range(c->arg_min, c->arg_max);
      // Only chains read past mem[0]; filling just what is used keeps xsim runs short.
      const int n_mem = arg > 1 && c->arg_min > 0 ? arg : 1;
      for (int k = 0; k < n_mem; k++) fill_vector(&mem[k], bits);
      if (c->mem_kind == MEM_SHIFTS) fill_shifts(&mem[0], mode);

      hw_run(mode, &in, mem, c->hw, arg, &hw);
      sim_run(mode, &in, mem, c->sim, arg, &sim);

      if (memcmp(&hw, &sim, sizeof(hw)) != 0) {
        if (fails < MAX_REPORTED) report(c, mode, arg, &in, mem, &hw, &sim);
        fails++;
      }
    }
    printf("  %-12s %-4s %4u/%u mismatches\n", c->name, mode_name(mode), fails,
           VPU_SIM_RANDOM_ITERS);
    total += fails;
  }
  return total;
}

// ---- operations: hardware ----
HW_OP_FPTR static void hw_vclrdr(const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; vclrdr(); }
HW_OP_FPTR static void hw_vlmacc(const vpu_vector_t *mem, int arg) { for (int k = 0; k < arg; k++) vlmacc(&mem[k]); }
HW_OP_FPTR static void hw_vlmaccr(const vpu_vector_t *mem, int arg) { for (int k = 0; k < arg; k++) vlmaccr(&mem[k]); }
HW_OP_FPTR static void hw_vlmaccr1(const vpu_vector_t *mem, int arg) { for (int k = 0; k < arg; k++) vlmaccr_binary(&mem[k]); }
HW_OP_FPTR static void hw_vpos(const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; vpos(); }
HW_OP_FPTR static void hw_vlsat(const vpu_vector_t *mem, int arg) { (void)arg; vlsat(mem); }
HW_OP_FPTR static void hw_vlashr(const vpu_vector_t *mem, int arg) { vlashr(mem, (int8_t)arg); }
HW_OP_FPTR static void hw_vladd(const vpu_vector_t *mem, int arg) { (void)arg; vladd(mem); }
HW_OP_FPTR static void hw_vlsub(const vpu_vector_t *mem, int arg) { (void)arg; vlsub(mem); }
HW_OP_FPTR static void hw_vlmul(const vpu_vector_t *mem, int arg) { (void)arg; vlmul(mem); }
HW_OP_FPTR static void hw_vdepth1(const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; vdepth1(); }
HW_OP_FPTR static void hw_vdepth8(const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; vdepth8(); }
HW_OP_FPTR static void hw_vdepth16(const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; vdepth16(); }

// ---- operations: model ----
SIM_OP_FPTR static void sim_vclrdr(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; VCLRDR(v); }
SIM_OP_FPTR static void sim_vlmacc(xs3_vpu *v, const vpu_vector_t *mem, int arg) { for (int k = 0; k < arg; k++) VLMACC(v, &mem[k]); }
SIM_OP_FPTR static void sim_vlmaccr(xs3_vpu *v, const vpu_vector_t *mem, int arg) { for (int k = 0; k < arg; k++) VLMACCR(v, &mem[k]); }
SIM_OP_FPTR static void sim_vlmaccr1(xs3_vpu *v, const vpu_vector_t *mem, int arg) { for (int k = 0; k < arg; k++) VLMACCR1(v, &mem[k]); }
SIM_OP_FPTR static void sim_vpos(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; VPOS(v); }
SIM_OP_FPTR static void sim_vlsat(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)arg; VLSAT(v, mem); }
SIM_OP_FPTR static void sim_vlashr(xs3_vpu *v, const vpu_vector_t *mem, int arg) { VLASHR(v, mem, arg); }
SIM_OP_FPTR static void sim_vladd(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)arg; VLADD(v, mem); }
SIM_OP_FPTR static void sim_vlsub(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)arg; VLSUB(v, mem); }
SIM_OP_FPTR static void sim_vlmul(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)arg; VLMUL(v, mem); }
SIM_OP_FPTR static void sim_vdepth1(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; VDEPTH1(v); }
SIM_OP_FPTR static void sim_vdepth8(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; VDEPTH8(v); }
SIM_OP_FPTR static void sim_vdepth16(xs3_vpu *v, const vpu_vector_t *mem, int arg) { (void)mem; (void)arg; VDEPTH16(v); }

static const unsigned ALL_MODES[] = {MODE_S8, MODE_S16, MODE_S32};
static const unsigned DEPTH8_MODES[] = {MODE_S16, MODE_S32};
static const unsigned S32_ONLY[] = {MODE_S32};

#define RUN_ALL(c) TEST_ASSERT_EQUAL_UINT(0, run_case(&(c), ALL_MODES, 3))

TEST_GROUP(group_vpu_sim_random);
TEST_SETUP(group_vpu_sim_random) {}
TEST_TEAR_DOWN(group_vpu_sim_random) {}

TEST(group_vpu_sim_random, vclrdr) {
  const vpu_case_t c = {"vclrdr", hw_vclrdr, sim_vclrdr, MEM_DATA, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlmacc) {
  const vpu_case_t c = {"vlmacc", hw_vlmacc, sim_vlmacc, MEM_DATA, 1, MAX_CHAIN};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlmaccr) {
  const vpu_case_t c = {"vlmaccr", hw_vlmaccr, sim_vlmaccr, MEM_DATA, 1, MAX_CHAIN};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlmaccr1) {
  // The model only implements the binary MAC in the 8/16-bit modes lib_nn uses it in.
  const vpu_case_t c = {"vlmaccr1", hw_vlmaccr1, sim_vlmaccr1, MEM_DATA, 1, MAX_CHAIN};
  TEST_ASSERT_EQUAL_UINT(0, run_case(&c, ALL_MODES, 2));
}

TEST(group_vpu_sim_random, vpos) {
  const vpu_case_t c = {"vpos", hw_vpos, sim_vpos, MEM_DATA, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlsat) {
  const vpu_case_t c = {"vlsat", hw_vlsat, sim_vlsat, MEM_SHIFTS, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlashr) {
  const vpu_case_t c = {"vlashr", hw_vlashr, sim_vlashr, MEM_DATA, -40, 40};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vladd) {
  const vpu_case_t c = {"vladd", hw_vladd, sim_vladd, MEM_DATA, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlsub) {
  const vpu_case_t c = {"vlsub", hw_vlsub, sim_vlsub, MEM_DATA, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vlmul) {
  const vpu_case_t c = {"vlmul", hw_vlmul, sim_vlmul, MEM_DATA, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vdepth1) {
  const vpu_case_t c = {"vdepth1", hw_vdepth1, sim_vdepth1, MEM_DATA, 0, 0};
  RUN_ALL(c);
}

TEST(group_vpu_sim_random, vdepth8) {
  const vpu_case_t c = {"vdepth8", hw_vdepth8, sim_vdepth8, MEM_DATA, 0, 0};
  TEST_ASSERT_EQUAL_UINT(0, run_case(&c, DEPTH8_MODES, 2));
}

TEST(group_vpu_sim_random, vdepth16) {
  const vpu_case_t c = {"vdepth16", hw_vdepth16, sim_vdepth16, MEM_DATA, 0, 0};
  TEST_ASSERT_EQUAL_UINT(0, run_case(&c, S32_ONLY, 1));
}

TEST(group_vpu_sim_random, vstrpv) {
  unsigned fails = 0;
  for (unsigned m = 0; m < 3; m++) {
    const unsigned mode = ALL_MODES[m];
    for (unsigned it = 0; it < VPU_SIM_RANDOM_ITERS; it++) {
      vpu_vector_t WORD_ALIGNED in, out_hw, out_sim;
      xs3_vpu vpu;
      fill_vector(&in, mode_bits(mode));
      fill_vector(&out_hw, 8);
      memcpy(&out_sim, &out_hw, sizeof(out_sim));
      const unsigned mask = pseudo_rand_uint32();

      vsetc_reg(mode);
      vldr(&in);
      vstrpv(&out_hw, mask);

      memset(&vpu, 0, sizeof(vpu));
      VSETC(&vpu, (vector_mode)mode);
      VLDR(&vpu, &in);
      VSTRPV(&vpu, &out_sim, mask);

      if (memcmp(&out_hw, &out_sim, sizeof(out_hw)) != 0) {
        if (fails < MAX_REPORTED) {
          printf("  MISMATCH vstrpv %s mask=%08X\n", mode_name(mode), mask);
          print_words("hw", &out_hw);
          print_words("sim", &out_sim);
        }
        fails++;
      }
    }
  }
  printf("  %-12s all  %4u/%u mismatches\n", "vstrpv", fails, 3 * VPU_SIM_RANDOM_ITERS);
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

TEST_GROUP_RUNNER(group_vpu_sim_random) {
  RUN_TEST_CASE(group_vpu_sim_random, vclrdr);
  RUN_TEST_CASE(group_vpu_sim_random, vstrpv);
  RUN_TEST_CASE(group_vpu_sim_random, vlmacc);
  RUN_TEST_CASE(group_vpu_sim_random, vlmaccr);
  RUN_TEST_CASE(group_vpu_sim_random, vlmaccr1);
  RUN_TEST_CASE(group_vpu_sim_random, vpos);
  RUN_TEST_CASE(group_vpu_sim_random, vlsat);
  RUN_TEST_CASE(group_vpu_sim_random, vlashr);
  RUN_TEST_CASE(group_vpu_sim_random, vladd);
  RUN_TEST_CASE(group_vpu_sim_random, vlsub);
  RUN_TEST_CASE(group_vpu_sim_random, vlmul);
  RUN_TEST_CASE(group_vpu_sim_random, vdepth1);
  RUN_TEST_CASE(group_vpu_sim_random, vdepth8);
  RUN_TEST_CASE(group_vpu_sim_random, vdepth16);
}

#endif  // TEST_BUILD_NATIVE
