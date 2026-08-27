// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdio.h>

#include "unity_fixture.h"

int main(int argc, const char* argv[]) {
  printf("Running integration tests for lib_nn\n");
  UnityGetCommandLineOptions(argc, argv);
  UnityBegin(argv[0]);

  RUN_TEST_GROUP(group_Conv2dRegression);

  return UnityEnd();
}
