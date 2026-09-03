// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#pragma once

#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif

#define C_API EXTERN_C

#define ERR_MSG_DESCRIPTOR_FAIL_BYTES() (128)

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__xcore__) || defined(__riscv_xxcore)
#define WORD_ALIGNED __attribute__((aligned(4)))
#else
#define WORD_ALIGNED
#endif
