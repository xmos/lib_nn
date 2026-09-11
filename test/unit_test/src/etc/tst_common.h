// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#ifndef TST_COMMON_H_
#define TST_COMMON_H_

#include <stdint.h>
#include <nn_api.h>

#ifdef __XC__
extern "C" {
#endif

int8_t pseudo_rand_int8();
int16_t pseudo_rand_int16();
uint16_t pseudo_rand_uint16();
int32_t pseudo_rand_int32();
uint32_t pseudo_rand_uint32();
int64_t pseudo_rand_int64();
uint64_t pseudo_rand_uint64();

void pseudo_rand_bytes(char* buffer, unsigned size);

void print_warns(int start_case);

#ifdef __XC__
}  // extern "C"
#endif

#endif  // TST_COMMON_H_
