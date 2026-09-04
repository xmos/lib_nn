// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdio.h>

#include "unity_fixture.h"

int main(int argc, const char* argv[]) {
  printf("Running unit tests for lib_nn\n");
  UnityGetCommandLineOptions(argc, argv);
  UnityBegin(argv[0]);

  RUN_TEST_GROUP(group_add_elementwise);
  RUN_TEST_GROUP(group_add_int16);
  RUN_TEST_GROUP(group_bsign_8);
  RUN_TEST_GROUP(group_dequantize_int16);
  RUN_TEST_GROUP(group_expand_8_to_16);
  RUN_TEST_GROUP(group_lookup8);
  RUN_TEST_GROUP(group_matmul);
  RUN_TEST_GROUP(group_mul_elementwise);
  RUN_TEST_GROUP(group_multiply_int16);
  RUN_TEST_GROUP(group_output_transform_fn_int16);
  RUN_TEST_GROUP(group_pad_3_to_4);
  RUN_TEST_GROUP(group_quantize_int16);
  RUN_TEST_GROUP(group_quadratic_interpolation);
  RUN_TEST_GROUP(group_softmax);
  RUN_TEST_GROUP(group_vpu);

  // -------- Native only --------------
  #ifdef TEST_BUILD_NATIVE
    RUN_TEST_GROUP(group_aggregate_fns);
    RUN_TEST_GROUP(group_Filter2dGeometry);
    RUN_TEST_GROUP(group_ImageGeometry);
    RUN_TEST_GROUP(group_ImageRegion);
    RUN_TEST_GROUP(group_ImageVect);
    RUN_TEST_GROUP(group_maxpool);
    RUN_TEST_GROUP(group_mem_cpy_fns);
    RUN_TEST_GROUP(group_output_transforms);
    RUN_TEST_GROUP(group_WindowGeometry);
    RUN_TEST_GROUP(group_WindowLocation);
  #endif
  
  return UnityEnd();
}
