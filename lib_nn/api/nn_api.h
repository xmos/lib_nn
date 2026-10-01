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
#if defined(__XS3A__) && defined(__riscv_xxcore)
#error "lib_nn target cannot use both legacy XCORE and RISC-V instruction sets"
#endif

/**
 * @def VPU_SYMMETRIC_SATURATION
 * @brief Indicates whether signed VPU saturation uses symmetric bounds.
 *
 * A value of 1 selects symmetric bounds such as [-127, 127] for 8-bit values;
 * 0 selects the full two's-complement range such as [-128, 127].
 */
#ifndef VPU_SYMMETRIC_SATURATION
#if defined(__XS3A__)
#define VPU_SYMMETRIC_SATURATION 1
#elif defined(__riscv_xxcore)
#define VPU_SYMMETRIC_SATURATION 0
#endif
#endif

/**
 * @def VPU_VLMACC_PRODUCT_LSB_DROPS
 * @brief Number of least-significant product bits discarded by VLMACC.
 *
 * This capability describes product precision before accumulation and is
 * independent of the instruction-set spelling.
 */
#ifndef VPU_VLMACC_PRODUCT_LSB_DROPS
#if defined(__XS3A__)
#define VPU_VLMACC_PRODUCT_LSB_DROPS 0
#elif defined(__riscv_xxcore)
#define VPU_VLMACC_PRODUCT_LSB_DROPS 1
#endif
#endif

/**
 * @def VPU_VLMUL_SHIFT_OFFSET
 * @brief Offset subtracted from the element width to obtain the VLMUL shift.
 *
 * For an element width of `N` bits, VLMUL scales its product using a shift of
 * `N - VPU_VLMUL_SHIFT_OFFSET` bits.
 */
#ifndef VPU_VLMUL_SHIFT_OFFSET
#if defined(__XS3A__)
#define VPU_VLMUL_SHIFT_OFFSET 2
#elif defined(__riscv_xxcore)
#define VPU_VLMUL_SHIFT_OFFSET 1
#endif
#endif

/**
 * @def VPU_CONFIGURED
 * @brief Indicates whether all required VPU behavior capabilities are defined.
 *
 * This derived macro is 1 only when `VPU_SYMMETRIC_SATURATION`,
 * `VPU_VLMACC_PRODUCT_LSB_DROPS`, and `VPU_VLMUL_SHIFT_OFFSET` are all
 * available; otherwise it is 0.
 */
#if defined(VPU_SYMMETRIC_SATURATION) && \
	defined(VPU_VLMACC_PRODUCT_LSB_DROPS) && \
	defined(VPU_VLMUL_SHIFT_OFFSET)
#define VPU_CONFIGURED 1
#else
#define VPU_CONFIGURED 0
#endif

#if defined(__riscv_xxcore) && !VPU_CONFIGURED
#error "RISC-V builds must define the VPU_* capability macros"
#endif

/**
 * @def HAS_FPTRGROUP
 * @brief Indicates whether the compiler supports the `fptrgroup` attribute.
 *
 * Code that groups indirect-call targets may use this capability to omit the
 * attribute on unsupported compilers.
 */
#ifndef HAS_FPTRGROUP
#if defined(__XS3A__) || defined(__riscv_xxcore)
#define HAS_FPTRGROUP 1
#else
#define HAS_FPTRGROUP 0
#endif
#endif

/**
 * @def HAS_FULL_CXX_RTTI
 * @brief Indicates whether the C++ runtime provides full RTTI support.
 *
 * A value of 0 identifies reduced runtimes that cannot provide all typeinfo
 * objects required by some polymorphic standard-library facilities.
 */
#ifndef HAS_FULL_CXX_RTTI
#if defined(__VX4A__) || defined(__riscv_xxcore)
#define HAS_FULL_CXX_RTTI 0
#else
#define HAS_FULL_CXX_RTTI 1
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
