// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "unity_fixture.h"

#include "vpu_sim.h"
#include "xs3_vpu.h"
#include "nn_arch.h"

#include "tst_common.h"
#include "etc/test_vpu_sim.h"

TEST_GROUP(group_vpu_sim);
TEST_SETUP(group_vpu_sim) {
#if defined(__VX4A__) || defined(__VX4B__)
  SetNNTargetArch(TARGET_ARCH_VX4A);
#else
  SetNNTargetArch(TARGET_ARCH_XS3A);
#endif
}
TEST_TEAR_DOWN(group_vpu_sim) {}
TEST_GROUP_RUNNER(group_vpu_sim) {
  RUN_TEST_CASE(group_vpu_sim, test_basic);
  RUN_TEST_CASE(group_vpu_sim, test_vstrpv);
  
  RUN_TEST_CASE(group_vpu_sim, test_vlmacc);
  RUN_TEST_CASE(group_vpu_sim, test_vlmaccr);

  RUN_TEST_CASE(group_vpu_sim, test_vl_add_sub_mul);
  
  RUN_TEST_CASE(group_vpu_sim, test_vpos);
  RUN_TEST_CASE(group_vpu_sim, test_vlashr);
  RUN_TEST_CASE(group_vpu_sim, test_vdepth1);
  RUN_TEST_CASE(group_vpu_sim, test_vdepth8);
  RUN_TEST_CASE(group_vpu_sim, test_vdepth16);
}

TEST(group_vpu_sim, test_basic) {
  // simply load/clear/save vectors, expect r/d to clear and c to persist
  // hw and sim are expected to match
    int8_t WORD_ALIGNED vr[VPU_INT8_EPV] = {1};
    int8_t WORD_ALIGNED vd[VPU_INT8_EPV] = {-1};
    int8_t WORD_ALIGNED vc[VPU_INT8_EPV] = {2};
    int8_t WORD_ALIGNED expected[VPU_INT8_EPV] = {0};
    int8_t WORD_ALIGNED expected_vc[VPU_INT8_EPV] = {2};

    vsetc(MODE_S8);
    
    vldr(vr); vldd(vd); vldc(vc); 
    vclrdr();
    vstr(vr); vstd(vd); vstc(vc);

    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, vr, VPU_INT8_EPV);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, vd, VPU_INT8_EPV);
    TEST_ASSERT_EQUAL_INT8_ARRAY(expected_vc, vc, VPU_INT8_EPV);
}

TEST(group_vpu_sim, test_vstrpv) {
  // different masks are tested with different inputs
  // hw and sim are expected to match
  const unsigned n_runs = 10;

  unsigned mask = 0xAAAAAAAA;
  int8_t input_value = 0;
  int8_t output_value = 0;

  int8_t WORD_ALIGNED input[VPU_INT8_EPV];
  int8_t WORD_ALIGNED out_asm[VPU_INT8_EPV];
  int8_t WORD_ALIGNED out_sim[VPU_INT8_EPV];
  vpu_t sim = {0};

  // First run: fixed input for a simple, predictable baseline.
  memset(input, 1, sizeof(input));
  memset(out_asm, 0x55, sizeof(out_asm));
  memset(out_sim, 0x55, sizeof(out_sim));

  // xmos assembly
  vsetc(MODE_S8);
  vldr(input);
  vstrpv(out_asm, mask);

  // simulated vpu
  VSETC(&sim, MODE_S8);
  VLDR(&sim, input);
  VSTRPV(&sim, out_sim, mask);

  TEST_ASSERT_EQUAL_INT8_ARRAY(out_sim, out_asm, VPU_INT8_EPV);

  // Remaining runs: vary the value written to every input element.
  for (unsigned run = 1; run < n_runs; ++run) {
    input_value = pseudo_rand_int8();
    output_value = pseudo_rand_int8();
    mask = pseudo_rand_uint32();
    memset(input, input_value, sizeof(input));
    memset(out_asm, output_value, sizeof(out_asm));
    memset(out_sim, output_value, sizeof(out_sim));

    vsetc(MODE_S8);
    vldr(input);
    vstrpv(out_asm, mask);
    
    VSETC(&sim, MODE_S8);
    VLDR(&sim, input);
    VSTRPV(&sim, out_sim, mask);
    
    TEST_ASSERT_EQUAL_INT8_ARRAY(out_sim, out_asm, VPU_INT8_EPV);
  }
}

TEST(group_vpu_sim, test_vlmacc) {
  // multiply two vectors and saturate the accumulated result
  // hw and sim are expected to match
  int16_t WORD_ALIGNED coefficients[VPU_INT16_EPV];
  int16_t WORD_ALIGNED input[VPU_INT16_EPV];
  int16_t WORD_ALIGNED shifts[VPU_INT16_EPV];
  int16_t WORD_ALIGNED out_asm[VPU_INT16_EPV];
  int16_t WORD_ALIGNED out_sim[VPU_INT16_EPV];
  int16_t WORD_ALIGNED expected[VPU_INT16_EPV] = {
      258, 258, 258, 258, 258, 258, 258, 258,
      258, 258, 258, 258, 258, 258, 258, 258};
  vpu_t sim = {0};

  // memset fills bytes: 0xFFFF is -1 and 0xFEFE is -258 as int16_t.
  // vlmacc then should produce 258
  memset(coefficients, 0xFF, sizeof(coefficients));
  memset(input, 0xFE, sizeof(input));
  memset(shifts, 0, sizeof(shifts));

  vsetc(MODE_S16);
  vclrdr();
  vldc((const int8_t *)coefficients);
  vlmacc((const int8_t *)input);
  vlsat(shifts);
  vstr((int8_t *)out_asm);

  VSETC(&sim, MODE_S16);
  VCLRDR(&sim);
  VLDC(&sim, coefficients);
  VLMACC(&sim, input);
  VLSAT(&sim, shifts);
  VSTR(&sim, out_sim);

  TEST_ASSERT_EQUAL_INT16_ARRAY(expected, out_asm, VPU_INT16_EPV);
  TEST_ASSERT_EQUAL_INT16_ARRAY(expected, out_sim, VPU_INT16_EPV);
  TEST_ASSERT_EQUAL_INT16_ARRAY(out_sim, out_asm, VPU_INT16_EPV);
}

TEST(group_vpu_sim, test_vlmaccr) {
  // multiply and accumulate one complete vector into the ring buffer
  // hw and sim are expected to match
  int16_t WORD_ALIGNED coefficients[VPU_INT16_EPV];
  int16_t WORD_ALIGNED input[VPU_INT16_EPV];
  int16_t WORD_ALIGNED shifts[VPU_INT16_EPV];
  int16_t WORD_ALIGNED out_asm[VPU_INT16_EPV];
  int16_t WORD_ALIGNED out_sim[VPU_INT16_EPV];
  int16_t WORD_ALIGNED expected[VPU_INT16_EPV];
  vpu_t sim = {0};

  // Each of the 16 products is (-1) * (-258) = 258.
  memset(coefficients, 0xFF, sizeof(coefficients));
  memset(input, 0xFE, sizeof(input));
  memset(shifts, 0, sizeof(shifts));
  memset(expected, 0, sizeof(expected));
  expected[0] = 16 * 258;

  vsetc(MODE_S16);
  vclrdr();
  vldc((const int8_t *)coefficients);
  vlmaccr((const int8_t *)input);
  vlsat(shifts);
  vstr((int8_t *)out_asm);

  VSETC(&sim, MODE_S16);
  VCLRDR(&sim);
  VLDC(&sim, coefficients);
  VLMACCR(&sim, input);
  VLSAT(&sim, shifts);
  VSTR(&sim, out_sim);

  printf("vlmaccr out_asm[0] = 0x%04X, out_sim[0] = 0x%04X\n",
         (uint16_t)out_asm[0], (uint16_t)out_sim[0]);

  TEST_ASSERT_EQUAL_INT16_ARRAY(expected, out_asm, VPU_INT16_EPV);
  TEST_ASSERT_EQUAL_INT16_ARRAY(expected, out_sim, VPU_INT16_EPV);
  TEST_ASSERT_EQUAL_INT16_ARRAY(out_sim, out_asm, VPU_INT16_EPV);
}
TEST(group_vpu_sim, test_vpos) {}
TEST(group_vpu_sim, test_vlashr) {}
TEST(group_vpu_sim, test_vl_add_sub_mul) {
  // apply add, subtract, and multiply as one instruction sequence
  // hw and sim are expected to match
  int8_t WORD_ALIGNED a[VPU_INT8_EPV];
  int8_t WORD_ALIGNED b[VPU_INT8_EPV];
  int8_t WORD_ALIGNED c[VPU_INT8_EPV];
  int8_t WORD_ALIGNED d[VPU_INT8_EPV];
  int8_t WORD_ALIGNED out_asm[VPU_INT8_EPV];
  int8_t WORD_ALIGNED out_sim[VPU_INT8_EPV];
  int8_t WORD_ALIGNED expected_xs3[VPU_INT8_EPV];
  int8_t WORD_ALIGNED expected_vx4[VPU_INT8_EPV];
  vpu_t sim = {0};

  // (a + b - c) * d = (10 + 20 - 5) * 64 = 1600.
  // VLSUB computes c - (a + b) = -25 before VLMUL scaling.
  memset(a, 10, sizeof(a));
  memset(b, 20, sizeof(b));
  memset(c, 5, sizeof(c));
  memset(d, 64, sizeof(d));
  memset(expected_xs3, -25, sizeof(expected_xs3));
  memset(expected_vx4, -12, sizeof(expected_vx4));

  vsetc(MODE_S8);
  vldr(a);
  vladd(b);
  vlsub(c);
  vlmul(d);
  vstr(out_asm);

  VSETC(&sim, MODE_S8);
  VLDR(&sim, a);
  VLADD(&sim, b);
  VLSUB(&sim, c);
  VLMUL(&sim, d);
  VSTR(&sim, out_sim);

#if defined(__XS3A__)
  TEST_ASSERT_EQUAL_INT8_ARRAY(expected_xs3, out_asm, VPU_INT8_EPV);
  TEST_ASSERT_EQUAL_INT8_ARRAY(expected_xs3, out_sim, VPU_INT8_EPV);
#elif defined(__VX4B__)
  TEST_ASSERT_EQUAL_INT8_ARRAY(expected_vx4, out_asm, VPU_INT8_EPV);
  TEST_ASSERT_EQUAL_INT8_ARRAY(expected_vx4, out_sim, VPU_INT8_EPV);
#endif
  TEST_ASSERT_EQUAL_INT8_ARRAY(out_sim, out_asm, VPU_INT8_EPV);
}

TEST(group_vpu_sim, test_vdepth1) {
  // reduce an S8 vector to a packed sign mask
  // hw and sim are expected to match
  int8_t WORD_ALIGNED input_s8[VPU_INT8_EPV] = {
    -1, 0, 1, -2, 2, -3, 3, -4,
    4, -5, 5, -6, 6, -7, 7, -8,
    8, -9, 9, -10, 10, -11, 11, -12,
    12, -13, 13, -14, 14, -15, 15, 16
  };
  int8_t WORD_ALIGNED out_asm[XS3_VPU_VREG_WIDTH_BYTES];
  int8_t WORD_ALIGNED out_sim[XS3_VPU_VREG_WIDTH_BYTES];
  vpu_t sim = {0};

  vsetc(MODE_S8);
  vldr(input_s8);
  vdepth1();
  vstr(out_asm);
  VSETC(&sim, MODE_S8);
  VLDR(&sim, input_s8);
  VDEPTH1(&sim);
  VSTR(&sim, out_sim);
  TEST_ASSERT_EQUAL_INT8_ARRAY(out_sim, out_asm, sizeof(out_asm));
}

TEST(group_vpu_sim, test_vdepth8) {
  // reduce an S16 vector to S8 with rounding and saturation
  // hw and sim are expected to match
  int16_t WORD_ALIGNED input_s16[VPU_INT16_EPV] = {
    -32768, -32767, -256, -255, -1, 0, 255, 256,
    32766, 32767, 128, -128, 384, -384, 1024, -1024
  };
  int8_t WORD_ALIGNED out_asm[XS3_VPU_VREG_WIDTH_BYTES];
  int8_t WORD_ALIGNED out_sim[XS3_VPU_VREG_WIDTH_BYTES];
  vpu_t sim = {0};

  memset(out_asm, 0, sizeof(out_asm));
  memset(out_sim, 0, sizeof(out_sim));
  vsetc(MODE_S16);
  vldr((const int8_t *)input_s16);
  vdepth8();
  vstr(out_asm);
  VSETC(&sim, MODE_S16);
  VLDR(&sim, input_s16);
  VDEPTH8(&sim);
  VSTR(&sim, out_sim);
  TEST_ASSERT_EQUAL_INT8_ARRAY(out_sim, out_asm, sizeof(out_asm));
}

TEST(group_vpu_sim, test_vdepth16) {
  // reduce an S32 vector to S16 with rounding and saturation
  // hw and sim are expected to match
  int32_t WORD_ALIGNED input_s32[VPU_INT32_EPV] = {
    -2147483647, -2147450880, -65536, -65535,-1, 0, 65535, 65536
  };
  int8_t WORD_ALIGNED out_asm[XS3_VPU_VREG_WIDTH_BYTES];
  int8_t WORD_ALIGNED out_sim[XS3_VPU_VREG_WIDTH_BYTES];
  vpu_t sim = {0};

  memset(out_asm, 0, sizeof(out_asm));
  memset(out_sim, 0, sizeof(out_sim));
  vsetc(MODE_S32);
  vldr((const int8_t *)input_s32);
  vdepth16();
  vstr(out_asm);
  VSETC(&sim, MODE_S32);
  VLDR(&sim, input_s32);
  VDEPTH16(&sim);
  VSTR(&sim, out_sim);
  TEST_ASSERT_EQUAL_INT8_ARRAY(out_sim, out_asm, sizeof(out_asm));
}
