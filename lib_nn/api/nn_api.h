// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

/**
 * @file nn_api.h
 * @brief Common XCORE target capabilities and portable declaration helpers.
 *
 * This header describes instruction-set-independent VPU behavior, compiler and
 * runtime capabilities, and declaration helpers used by lib_nn. These symbols
 * apply across the XCORE ecosystem and are expected to move to a shared
 * repository. They remain here temporarily so applications and libraries use
 * one consistent target description.
 *
 * Applications may override macros protected by `#ifndef` before including
 * this header. Other macros are derived from the selected target and language.
 */
#pragma once

/* Compiler-provided instruction-set macros must be mutually exclusive. */
#if defined(__xcore__) && defined(__riscv_xxcore)
#error "lib_nn target cannot use both legacy XCORE and RISC-V instruction sets"
#endif

/* Remove when the VX Clang toolchain defines __VX4B__ itself. */
#if defined(__riscv_xxcore) && !defined(__VX4B__)
#define __VX4B__ 1
#endif

/**
 * @def HAS_FPTRGROUP
 * @brief Indicates whether the compiler supports the `fptrgroup` attribute.
 *
 * Code that groups indirect-call targets may use this capability to omit the
 * attribute on unsupported compilers.
 */
#ifndef HAS_FPTRGROUP
#if defined(__xcore__) || defined(__riscv_xxcore)
#define HAS_FPTRGROUP 1
#else
#define HAS_FPTRGROUP 0
#endif
#endif

/**
 * @def C_API
 * @brief Gives a function C language linkage when compiled as C++ or XC.
 *
 * The macro expands to `extern "C"` for C++ and XC and to nothing for C.
 */
#if defined(__cplusplus) || defined(__XC__)
#define C_API extern "C"
#else
#define C_API
#endif

/**
 * @def ERR_MSG_DESCRIPTOR_FAIL_BYTES()
 * @brief Returns the reserved byte count for descriptor failure messages.
 */
#define ERR_MSG_DESCRIPTOR_FAIL_BYTES() (128)

/**
 * @def __has_builtin(x)
 * @brief Tests whether the compiler provides the named builtin.
 *
 * Compilers without their own `__has_builtin` operator receive a fallback
 * which evaluates to 0 for every argument.
 */
#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

/**
 * @def WORD_ALIGNED
 * @brief Requests four-byte alignment when supported by the XCORE compiler.
 *
 * The macro expands to an alignment attribute for XCORE builds and to nothing
 * for other targets.
 */
#if defined(__xcore__)
#define WORD_ALIGNED __attribute__((aligned(4)))
#else
#define WORD_ALIGNED
#endif

/**
 * @def DWORD_ALIGNED
 * @brief Requests eight-byte alignment when supported by the XCORE compiler.
 *
 * The macro expands to an alignment attribute for XCORE builds and to nothing
 * for other targets.
 */
#if defined(__xcore__)
#define DWORD_ALIGNED __attribute__((aligned(8)))
#else
#define DWORD_ALIGNED
#endif
