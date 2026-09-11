// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef TEST_QUADRATIC_APPROXIMATION_TABLES_H_
#define TEST_QUADRATIC_APPROXIMATION_TABLES_H_

#include <stdint.h>

#define TEST_QUADRATIC_APPROXIMATION_TABLE_COUNT 3
#define TEST_QUADRATIC_APPROXIMATION_TABLE_CHUNKS 128
#define TEST_QUADRATIC_APPROXIMATION_TABLE_BYTES \
    (TEST_QUADRATIC_APPROXIMATION_TABLE_CHUNKS * 8)

uint8_t *test_quadratic_approximation_table(unsigned index);

#endif
