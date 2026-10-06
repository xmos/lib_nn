// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

// Checks the scalar VPU models in OutputTransformFnInt8 (sat, shr, ashr, add, mul and
// quantised_output) bit-exactly against the hardware: each helper against the instruction it
// models, and quantised_output against the int8 output transform kernels.

#include <cstdint>
#include <cstdio>
#include <cstring>

#include "OutputTransformFn.hpp"

extern "C" {
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"
#include "vpu_sim.h"
}

#ifndef TEST_BUILD_NATIVE
#include "etc/test_vpu_sim.h"

#ifndef OT_HELPERS_ITERS
#define OT_HELPERS_ITERS 100
#endif
#define MAX_REPORTED 5

using nn::OutputTransformFnInt8;

typedef union {
  int16_t s16[VPU_INT16_EPV];
  uint16_t u16[VPU_INT16_EPV];
  int32_t s32[VPU_INT32_EPV];
} vec16_t;

static nn_vlmul_shr_t target_vlmul_shr(void) {
  return (NN_ARCH == TARGET_ARCH_XS3A) ? VLMUL_SHR_XS3A : VLMUL_SHR_VX4A;
}

static int rand_range(int lo, int hi) {
  return lo + (int)(pseudo_rand_uint32() % (uint32_t)(hi - lo + 1));
}

/** Random value of the given width: uniform, small, or an edge value. */
static int32_t rand_value(unsigned bits) {
  const int32_t max = (bits == 32) ? INT32_MAX : (1 << (bits - 1)) - 1;
  switch (pseudo_rand_uint32() % 8) {
    case 0: {
      const int32_t edges[] = {0, 1, -1, max, -max, -max - 1};
      return edges[pseudo_rand_uint32() % 6];
    }
    case 1:
    case 2:
    case 3:
      return (int32_t)pseudo_rand_uint32() >> (32 - bits + rand_range(0, bits - 1));
    default:
      return (int32_t)pseudo_rand_uint32() >> (32 - bits);
  }
}

static void fill16(vec16_t *v) {
  for (int i = 0; i < VPU_INT16_EPV; i++) v->s16[i] = (int16_t)rand_value(16);
}

static unsigned report(unsigned fails, const char *what, int lane, long long a, long long b,
                       long long arg, long long hw, long long model) {
  if (fails < MAX_REPORTED)
    printf("  MISMATCH %s lane %d: a=%lld b=%lld arg=%lld hw=%lld model=%lld\n", what, lane, a, b,
           arg, hw, model);
  return fails + 1;
}

extern "C" {

TEST_GROUP(group_output_transform_helpers);
TEST_SETUP(group_output_transform_helpers) {}
TEST_TEAR_DOWN(group_output_transform_helpers) {}

TEST(group_output_transform_helpers, add_vs_vladd) {
  const nn_vlmul_shr_t t = target_vlmul_shr();
  unsigned fails = 0;
  for (int it = 0; it < OT_HELPERS_ITERS; it++) {
    vec16_t WORD_ALIGNED a, b, out;
    fill16(&a);
    fill16(&b);
    vsetc(MODE_S16);
    vldr(&a);
    vladd(&b);
    vstr(&out);
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      const int32_t model = OutputTransformFnInt8::add(a.s16[i], b.s16[i], 16, t);
      if (out.s16[i] != model) fails = report(fails, "add", i, a.s16[i], b.s16[i], 0, out.s16[i], model);
    }
  }
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

TEST(group_output_transform_helpers, ashr_vs_vlashr) {
  const nn_vlmul_shr_t t = target_vlmul_shr();
  unsigned fails = 0;
  for (int it = 0; it < OT_HELPERS_ITERS; it++) {
    vec16_t WORD_ALIGNED a, out;
    fill16(&a);
    const int shift = rand_range(-16, 16);
    vsetc(MODE_S16);
    vlashr(&a, (int8_t)shift);
    vstr(&out);
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      const int32_t model = OutputTransformFnInt8::ashr(a.s16[i], shift, 16, t);
      if (out.s16[i] != model) fails = report(fails, "ashr", i, a.s16[i], 0, shift, out.s16[i], model);
    }
  }
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

TEST(group_output_transform_helpers, mul_vs_vlmul) {
  const nn_vlmul_shr_t t = target_vlmul_shr();
  unsigned fails = 0;
  for (int it = 0; it < OT_HELPERS_ITERS; it++) {
    vec16_t WORD_ALIGNED a, b, out;
    fill16(&a);
    fill16(&b);
    vsetc(MODE_S16);
    vldr(&a);
    vlmul(&b);
    vstr(&out);
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      const int32_t model = OutputTransformFnInt8::mul(a.s16[i], b.s16[i], 16, t);
      if (out.s16[i] != model) fails = report(fails, "mul", i, a.s16[i], b.s16[i], 0, out.s16[i], model);
    }
  }
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

TEST(group_output_transform_helpers, shr_vs_vlsat) {
  const nn_vlmul_shr_t t = target_vlmul_shr();
  unsigned fails = 0;
  for (int it = 0; it < OT_HELPERS_ITERS; it++) {
    // 32-bit accumulators are split across vD (high halves) and vR (low halves)
    int32_t acc[VPU_INT16_EPV];
    vec16_t WORD_ALIGNED hi, lo, shifts, out;
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      acc[i] = rand_value(32);
      hi.s16[i] = (int16_t)(acc[i] >> 16);
      lo.u16[i] = (uint16_t)acc[i];
      shifts.u16[i] = (uint16_t)rand_range(0, 24);
    }
    vsetc(MODE_S16);
    vldd(&hi);
    vldr(&lo);
    vlsat(&shifts);
    vstr(&out);
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      const int32_t model = OutputTransformFnInt8::shr(acc[i], shifts.u16[i], 16, t);
      if (out.s16[i] != model) fails = report(fails, "shr", i, acc[i], 0, shifts.u16[i], out.s16[i], model);
    }
  }
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

/** Random parameters for one channel of the int8 output transform. */
static void rand_channel(int32_t *acc, int16_t *mult, int16_t *bias) {
  *acc = rand_value(32) >> rand_range(0, 12);
  *mult = (int16_t)rand_value(16);
  *bias = (int16_t)rand_value(16);
}

TEST(group_output_transform_helpers, quantised_output_vs_otfn_int8) {
  const nn_vlmul_shr_t t = target_vlmul_shr();
  unsigned fails = 0;
  for (int it = 0; it < OT_HELPERS_ITERS; it++) {
    int32_t accs[VPU_INT16_EPV];
    int16_t params_mem[VPU_INT16_EPV * 2];
    int8_t out[VPU_INT16_EPV];
    VPURingBuffer acc{};
    for (int ch = 0; ch < VPU_INT16_EPV; ch++) {
      rand_channel(&accs[ch], &params_mem[ch], &params_mem[VPU_INT16_EPV + ch]);
      acc.SetAccu(ch, accs[ch]);
    }
    const int16_t initial_shift = (int16_t)rand_range(0, 16);
    const int16_t final_shr = (int16_t)rand_range(0, 8);

    nn::OT_int8 ot(VPU_INT16_EPV, initial_shift, final_shr);
    nn::otfn_int8_params_t params = ot.getParams();
    nn::otfn_int8(&params, out, &acc, 0, params_mem);

    for (int ch = 0; ch < VPU_INT16_EPV; ch++) {
      const int32_t model = OutputTransformFnInt8::quantised_output(
          accs[ch], initial_shift, params_mem[ch], params_mem[VPU_INT16_EPV + ch], final_shr, t);
      if (out[ch] != model) {
        if (fails < MAX_REPORTED)
          printf("  MISMATCH otfn_int8 ch %d: acc=%ld init=%d mult=%d bias=%d final=%d hw=%d model=%ld\n",
                 ch, (long)accs[ch], initial_shift, params_mem[ch], params_mem[VPU_INT16_EPV + ch],
                 final_shr, out[ch], (long)model);
        fails++;
      }
    }
  }
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

TEST(group_output_transform_helpers, quantised_output_vs_otfn_int8_channelwise) {
  const nn_vlmul_shr_t t = target_vlmul_shr();
  unsigned fails = 0;
  for (int it = 0; it < OT_HELPERS_ITERS; it++) {
    int32_t accs[VPU_INT16_EPV];
    // The channelwise layout is [initial shifts][multipliers][biases]
    int16_t params_mem[VPU_INT16_EPV * 3];
    int8_t out[VPU_INT16_EPV];
    VPURingBuffer acc{};
    for (int ch = 0; ch < VPU_INT16_EPV; ch++) {
      params_mem[ch] = (int16_t)rand_range(0, 16);
      rand_channel(&accs[ch], &params_mem[VPU_INT16_EPV + ch], &params_mem[2 * VPU_INT16_EPV + ch]);
      acc.SetAccu(ch, accs[ch]);
    }
    const int16_t final_shr = (int16_t)rand_range(0, 8);

    nn::OT_int8_channelwise ot(VPU_INT16_EPV, final_shr);
    nn::otfn_int8_channelwise_params_t params = ot.getParams();
    nn::otfn_int8_channelwise(&params, out, &acc, 0, params_mem);

    for (int ch = 0; ch < VPU_INT16_EPV; ch++) {
      const int32_t model = OutputTransformFnInt8::quantised_output(
          accs[ch], params_mem[ch], params_mem[VPU_INT16_EPV + ch],
          params_mem[2 * VPU_INT16_EPV + ch], final_shr, t);
      if (out[ch] != model) {
        if (fails < MAX_REPORTED)
          printf("  MISMATCH otfn_int8_channelwise ch %d: acc=%ld init=%d mult=%d bias=%d final=%d hw=%d model=%ld\n",
                 ch, (long)accs[ch], params_mem[ch], params_mem[VPU_INT16_EPV + ch],
                 params_mem[2 * VPU_INT16_EPV + ch], final_shr, out[ch], (long)model);
        fails++;
      }
    }
  }
  TEST_ASSERT_EQUAL_UINT(0, fails);
}

TEST_GROUP_RUNNER(group_output_transform_helpers) {
  RUN_TEST_CASE(group_output_transform_helpers, add_vs_vladd);
  RUN_TEST_CASE(group_output_transform_helpers, ashr_vs_vlashr);
  RUN_TEST_CASE(group_output_transform_helpers, mul_vs_vlmul);
  RUN_TEST_CASE(group_output_transform_helpers, shr_vs_vlsat);
  RUN_TEST_CASE(group_output_transform_helpers, quantised_output_vs_otfn_int8);
  RUN_TEST_CASE(group_output_transform_helpers, quantised_output_vs_otfn_int8_channelwise);
}

}  // extern "C"

#endif  // TEST_BUILD_NATIVE
