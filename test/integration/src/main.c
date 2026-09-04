// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdio.h>

#include "unity_fixture.h"

#define TEST_LEVEL_SMOKE 0
#define TEST_LEVEL_DEFAULT 1
#define TEST_LEVEL_EXTENDED 2

#ifndef TEST_LEVEL
#define TEST_LEVEL TEST_LEVEL_DEFAULT
#endif

int main(int argc, const char* argv[]) {
  printf("Running integration tests for lib_nn\n");
  UnityGetCommandLineOptions(argc, argv);
  UnityBegin(argv[0]);

  #if (TEST_LEVEL >= TEST_LEVEL_SMOKE)
  RUN_TEST_GROUP(group_Conv2dRegression);
  #endif

  #if (TEST_LEVEL >= TEST_LEVEL_DEFAULT)
  RUN_TEST_GROUP(group_Conv2dDenseReference);
  RUN_TEST_GROUP(group_TransposeConv2dRegression);
  #endif

  #if (TEST_LEVEL >= TEST_LEVEL_EXTENDED)
  RUN_TEST_GROUP(group_Conv2dRegression_DW);
  RUN_TEST_GROUP(group_Conv2dRegressionBinary);
  RUN_TEST_GROUP(group_Conv2dDenseBinaryReference);
  RUN_TEST_GROUP(group_Conv2dDepthwiseReference);
  #endif

  return UnityEnd();
}
