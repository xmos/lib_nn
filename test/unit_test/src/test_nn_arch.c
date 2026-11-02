// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "nn_arch.h"
#include "vpu_sim.h"

#include "unity_fixture.h"

TEST_GROUP(group_nn_arch);
TEST_SETUP(group_nn_arch) {}
TEST_TEAR_DOWN(group_nn_arch) {}

TEST_GROUP_RUNNER(group_nn_arch) {
  RUN_TEST_CASE(group_nn_arch, test_saturation_minimum);
}

TEST(group_nn_arch, test_saturation_minimum) {
  const nn_target_arch_t original_arch = NN_ARCH;

  SetNNTargetArch(TARGET_ARCH_XS3A);
  TEST_ASSERT_EQUAL_INT(-127, (int)vpu_saturate(-200, 8));
  TEST_ASSERT_TRUE(-((1LL << 39) - 1) == vpu_saturate(-(1LL << 40), 40));

  SetNNTargetArch(TARGET_ARCH_VX4A);
  TEST_ASSERT_EQUAL_INT(-128, (int)vpu_saturate(-200, 8));
  TEST_ASSERT_TRUE(-(1LL << 39) == vpu_saturate(-(1LL << 40), 40));

  SetNNTargetArch(original_arch);
}
