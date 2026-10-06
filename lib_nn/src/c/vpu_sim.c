// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "nn_arch.h"
#include "vpu_sim.h"

#include <stdio.h>
#include <stdlib.h>

#ifndef __cplusplus
#include <stdbool.h>
#endif

#ifdef _MSC_VER
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

/**
 * vpu_saturate to the relevant bounds.
 */
int64_t vpu_saturate(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  const int64_t min_val = -max_val;

  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

/**
 * vpu_saturate to the relevant bounds using the active target's minimum:
 * XS3 saturates symmetrically, VX4 uses the full two's complement range.
 * The target is chosen at runtime (NN_ARCH) so a host build can model either.
 */
int64_t vpu_saturate_fixed(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  const int64_t min_val = (NN_ARCH == TARGET_ARCH_VX4A) ? -max_val - 1 : -max_val;
  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

/** Arithmetic shift right by shr bits, rounding half up. */
static int64_t round_shr(const int64_t input, const unsigned shr) {
  if (shr == 0) return input;
  // Add the rounding bit after shifting so that values near INT64_MAX do not overflow
  return (input >> shr) + ((input >> (shr - 1)) & 1);
}

/**
 * Get the accumulator for the VPU's current mode
 */
static int64_t GetAccumulator(const xs3_vpu *vpu, unsigned index) {
  if (vpu->mode == MODE_S8 || vpu->mode == MODE_S16 || vpu->mode == MODE_S16x8) {
    union {
      int16_t s16[2];
      int32_t s32;
    } acc;
    acc.s16[1] = vpu->vD.s16[index];
    acc.s16[0] = vpu->vR.s16[index];
    return acc.s32;
  } else if (vpu->mode == MODE_S32) {
    // 40-bit accumulator: the low 32 bits in vR, 8 bits of headroom in the bottom of vD
    assert(index < VPU_INT32_EPV);
    return (int64_t)(int8_t)vpu->vD.s32[index] * ((int64_t)1 << 32) + vpu->vR.u32[index];
  } else {
    assert(0 && "Unsupported VPU mode");
  }
  return 0; // Should never reach here, avoid compiler warning
}

/**
 * Set the accumulator for the VPU's current mode
 */
static void SetAccumulator(xs3_vpu *vpu, unsigned index, int64_t acc) {
  if (vpu->mode == MODE_S8 || vpu->mode == MODE_S16 || vpu->mode == MODE_S16x8) {
    unsigned mask = (1 << VPU_INT8_ACC_VR_BITS) - 1;
    vpu->vR.s16[index] = (int16_t)((unsigned)acc & mask);
    mask = mask << VPU_INT8_ACC_VR_BITS;
    vpu->vD.s16[index] =
        (int16_t)(((unsigned)acc & mask) >> VPU_INT8_ACC_VR_BITS);
  } else if (vpu->mode == MODE_S32) {
    // The unused upper 24 bits of vD replicate the accumulator's sign
    assert(index < VPU_INT32_EPV);
    vpu->vR.u32[index] = (uint32_t)acc;
    vpu->vD.s32[index] = (int32_t)(acc >> 32);
  } else {
    assert(0);  // TODO
  }
}

/**
 * Rotate the accumulators following a VLMACCR
 */
static void rotate_accumulators(xs3_vpu *vpu) {
  if (vpu->mode == MODE_S8 || vpu->mode == MODE_S16 || vpu->mode == MODE_S16x8) {
    data16_t tmpD = vpu->vD.u16[VPU_INT8_ACC_PERIOD - 1];
    data16_t tmpR = vpu->vR.u16[VPU_INT8_ACC_PERIOD - 1];
    for (int i = VPU_INT8_ACC_PERIOD - 1; i > 0; i--) {
      vpu->vD.u16[i] = vpu->vD.u16[i - 1];
      vpu->vR.u16[i] = vpu->vR.u16[i - 1];
    }
    vpu->vD.u16[0] = tmpD;
    vpu->vR.u16[0] = tmpR;
  } else if (vpu->mode == MODE_S32) {
    uint32_t tmpD = vpu->vD.u32[VPU_INT32_ACC_PERIOD - 1];
    uint32_t tmpR = vpu->vR.u32[VPU_INT32_ACC_PERIOD - 1];
    for (int i = VPU_INT32_ACC_PERIOD - 1; i > 0; i--) {
      vpu->vD.u32[i] = vpu->vD.u32[i - 1];
      vpu->vR.u32[i] = vpu->vR.u32[i - 1];
    }
    vpu->vD.u32[0] = tmpD;
    vpu->vR.u32[0] = tmpR;
  } else {
    assert(0);  // How'd this happen?
  }
}

void VSETC(xs3_vpu *vpu, const vector_mode mode) { vpu->mode = mode; }

void VCLRDR(xs3_vpu *vpu) {
  memset(&vpu->vR.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
  memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDR(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  memcpy(&vpu->vR.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDD(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  memcpy(&vpu->vD.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDC(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  memcpy(&vpu->vC.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTR(const xs3_vpu *vpu, void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  memcpy(addr, &vpu->vR.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTD(const xs3_vpu *vpu, void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  memcpy(addr, &vpu->vD.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTC(const xs3_vpu *vpu, void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  memcpy(addr, &vpu->vC.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTRPV(const xs3_vpu *vpu, void *addr, unsigned mask) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  int8_t *addr8 = (int8_t *)addr;

  for (int i = 0; i < 32; i++) {
    if (mask & (1UL << i)) {
      addr8[i] = vpu->vR.s8[i];
    }
  }
}

void VLMACC(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;

    for (int i = 0; i < VPU_INT8_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + (((int32_t)vpu->vC.s8[i]) * addr8[i]);

      SetAccumulator(vpu, i, vpu_saturate_fixed(acc, 32));
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + (((int32_t)vpu->vC.s16[i]) * addr16[i]);
      // VX4 multiplies in two halves (VLMACC0/VLMACC1); the first halves the partial
      // sum, so the result loses its LSB.
      if (NN_ARCH == TARGET_ARCH_VX4A) acc = (acc >> 1) * 2;

      SetAccumulator(vpu, i, vpu_saturate_fixed(acc, 32));
    }
  } else if (vpu->mode == MODE_S16x8) {
    const int8_t *addr8 = (const int8_t *)addr;

    for (int i = 0; i < VPU_INT16_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + (((int32_t)vpu->vC.s16[i]) * (int16_t)(addr8[2*i]));

      SetAccumulator(vpu, i, vpu_saturate_fixed(acc, 32));
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + round_shr((int64_t)vpu->vC.s32[i] * addr32[i], 30);

      SetAccumulator(vpu, i, vpu_saturate_fixed(acc, 40));
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLMACCR(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    int64_t acc = GetAccumulator(vpu, VPU_INT8_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT8_EPV; i++)
      acc = acc + (((int32_t)vpu->vC.s8[i]) * addr8[i]);

    acc = vpu_saturate_fixed(acc, 32);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;
    int64_t acc = GetAccumulator(vpu, VPU_INT16_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT16_EPV; i++)
      acc = acc + (((int32_t)vpu->vC.s16[i]) * addr16[i]);
    // VX4 reduces in two halves (VLMACCR0/VLMACCR1); the first rounds the partial sum
    // to a multiple of 2, so an odd result is rounded up.
    if (NN_ARCH == TARGET_ARCH_VX4A) acc = ((acc + 1) >> 1) * 2;

    acc = vpu_saturate_fixed(acc, 32);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else if (vpu->mode == MODE_S16x8) {
    const int8_t *addr8 = (const int8_t *)addr;
    int64_t acc = GetAccumulator(vpu, VPU_INT16_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT16_EPV; i++)
      acc = acc + (((int32_t)vpu->vC.s16[i]) * (int16_t)(addr8[2*i]));

    acc = vpu_saturate_fixed(acc, 32);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;
    int64_t acc = GetAccumulator(vpu, VPU_INT32_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT32_EPV; i++)
      acc = acc + round_shr((int64_t)vpu->vC.s32[i] * addr32[i], 30);

    acc = vpu_saturate_fixed(acc, 40);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else {
    assert(0);  // How'd this happen?
  }
}

void VPOS(xs3_vpu *vpu) {
  // Operates on every element of vR, not on the accumulators
  if (vpu->mode == MODE_S8) {
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      if (vpu->vR.s8[i] < 0) vpu->vR.s8[i] = 0;
    }
  } else if (vpu->mode == MODE_S16) {
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      if (vpu->vR.s16[i] < 0) vpu->vR.s16[i] = 0;
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      if (vpu->vR.s32[i] < 0) vpu->vR.s32[i] = 0;
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLMACCR1(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  const int32_t *addr32 = (const int32_t *)addr;
  int64_t acc = GetAccumulator(vpu, VPU_BIN_ACC_PERIOD - 1);

  for (int i = 0; i < VPU_INT32_EPV; i++) {
    int v = (((int32_t)vpu->vC.s32[i]) ^ addr32[i]);
    acc += (2 * __builtin_popcount(~v) - 32) / 2;
  }

  acc = vpu_saturate_fixed(acc, 32);
  rotate_accumulators(vpu);
  SetAccumulator(vpu, 0, acc);
}

void VLMACCRB(xs3_vpu *vpu, const void *addr) {
  VLMACCR1(vpu, addr);
}

/**
 * VLSAT's rounding shift, for shift amounts that may exceed the accumulator width.
 * XS3 does not round once the shift reaches acc_bits; VX4 always rounds.
 */
static int64_t vlsat_shr(const int64_t acc, unsigned shr, const unsigned acc_bits) {
  if (shr > 62) shr = 62;
  if (NN_ARCH == TARGET_ARCH_XS3A && shr >= acc_bits) return acc >> shr;
  return round_shr(acc, shr);
}

/** Saturate to the full two's complement range, whatever the target. */
static int64_t saturate_asymmetric(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  return (input > max_val) ? max_val : (input < -max_val - 1) ? -max_val - 1 : input;
}

static void vlsat_impl(xs3_vpu *vpu, const void *addr, const bool asymmetric) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  if (vpu->mode == MODE_S8) {
    const uint16_t *addr16 = (const uint16_t *)addr;
    // VX4 produces 16-bit results in 8-bit mode (elements are max(bpe, 16) bits wide)
    const bool s16_results = !asymmetric && NN_ARCH == TARGET_ARCH_VX4A;

    for (int i = 0; i < VPU_INT8_ACC_PERIOD; i++) {
      int64_t acc = vlsat_shr(GetAccumulator(vpu, i), addr16[i], 32);
      if (s16_results)
        vpu->vR.s16[i] = (int16_t)vpu_saturate_fixed(acc, 16);
      else
        vpu->vR.s8[i] = (int8_t)(asymmetric ? saturate_asymmetric(acc, 8)
                                            : vpu_saturate_fixed(acc, 8));
    }
    if (!s16_results)
      memset(&vpu->vR.u8[VPU_INT8_ACC_PERIOD], 0, VPU_INT8_ACC_PERIOD);
    memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
  } else if (vpu->mode == MODE_S16) {
    const uint16_t *addr16 = (const uint16_t *)addr;

    for (int i = 0; i < VPU_INT16_ACC_PERIOD; i++) {
      int64_t acc = vlsat_shr(GetAccumulator(vpu, i), addr16[i], 32);
      vpu->vR.s16[i] = (int16_t)(asymmetric ? saturate_asymmetric(acc, 16)
                                            : vpu_saturate_fixed(acc, 16));
    }
    memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
  } else if (vpu->mode == MODE_S32) {
    const uint32_t *addr32 = (const uint32_t *)addr;

    for (int i = 0; i < VPU_INT32_ACC_PERIOD; i++) {
      // VLSAT reads all of vD, not just the 8 bits of headroom the MACs maintain
      int64_t acc = (int64_t)vpu->vD.s32[i] * ((int64_t)1 << 32) + vpu->vR.u32[i];
      acc = vlsat_shr(acc, addr32[i], 64);
      vpu->vR.s32[i] = (int32_t)(asymmetric ? saturate_asymmetric(acc, 32)
                                            : vpu_saturate_fixed(acc, 32));
    }
    memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLSAT(xs3_vpu *vpu, const void *addr) {
  vlsat_impl(vpu, addr, false);
}

/**
 * Not an instruction: VLSAT as the kernels' output sees it, saturating to the full two's
 * complement range on both targets. XS3 kernels fix up VLSAT's symmetric saturation themselves
 * (e.g. add_elementwise.S) and VX4 saturates this way natively. 8-bit results are packed into
 * the bottom 16 bytes of vR on both targets.
 */
void VLSAT_ASYMMETRIC(xs3_vpu *vpu, const void *addr) {
  vlsat_impl(vpu, addr, true);
}

/**
 * One element of VLASHR: an arithmetic shift right that does not round (a shift of bits or
 * more leaves only sign bits), or for negative shr a saturating shift left.
 */
static int64_t vlashr_element(const int64_t val, const int32_t shr, const unsigned bits) {
  // The result is always saturated, so on XS3 an unshifted MIN becomes -MAX
  if (shr >= (int32_t)bits) return vpu_saturate_fixed(val >> (bits - 1), bits);
  if (shr >= 0) return vpu_saturate_fixed(val >> shr, bits);
  // Shifting left by bits - 1 already saturates every non-zero value
  const unsigned shl = (-shr >= (int32_t)bits) ? bits - 1 : (unsigned)-shr;
  return vpu_saturate_fixed(val * ((int64_t)1 << shl), bits);
}

void VLASHR(xs3_vpu *vpu, const void *addr, const int32_t shr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++)
      vpu->vR.s8[i] = (int8_t)vlashr_element(addr8[i], shr, 8);
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;
    for (int i = 0; i < VPU_INT16_EPV; i++)
      vpu->vR.s16[i] = (int16_t)vlashr_element(addr16[i], shr, 16);
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;
    for (int i = 0; i < VPU_INT32_EPV; i++)
      vpu->vR.s32[i] = (int32_t)vlashr_element(addr32[i], shr, 32);
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLADD(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];
      vpu->vR.s8[i] = vpu_saturate_fixed((int32_t)vpu->vR.s8[i] + val, 8);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int32_t val = addr16[i];
      vpu->vR.s16[i] = vpu_saturate_fixed((int32_t)vpu->vR.s16[i] + val, 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      vpu->vR.s32[i] = vpu_saturate_fixed((int32_t)vpu->vR.s32[i] + val, 32);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

/**
 * One element of VLSUB, mem - vR. Subtracting MIN does not simply saturate: the hardware gives
 * MIN for 0 - MIN, and in 32-bit mode XS3 negates MIN to itself before adding.
 */
static int64_t vlsub_element(const int64_t mem, const int64_t r, const unsigned bits) {
  const int64_t min_val = -((int64_t)1 << (bits - 1));
  if (r == min_val) {
    if (NN_ARCH == TARGET_ARCH_XS3A && bits == 32) return vpu_saturate_fixed(mem + min_val, bits);
    if (mem == 0) return min_val;
  }
  return vpu_saturate_fixed(mem - r, bits);
}

void VLSUB(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];
      vpu->vR.s8[i] = vlsub_element(val, vpu->vR.s8[i], 8);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int32_t val = addr16[i];
      vpu->vR.s16[i] = vlsub_element(val, vpu->vR.s16[i], 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      vpu->vR.s32[i] = vlsub_element(val, vpu->vR.s32[i], 32);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

static inline
unsigned vlmul_get_shift(const nn_target_arch_t arch, const vector_mode mode) {
  // VLMUL shift = bpe - 2 for XS3A; bpe - 1 for VX4A in 8- and 16-bit modes, but 30 in
  // 32-bit mode
  assert(arch == TARGET_ARCH_XS3A || arch == TARGET_ARCH_VX4A);
  unsigned shift = 0;
  unsigned adj = (arch == TARGET_ARCH_XS3A) ? 0 : 1;
  switch (mode) {
    case MODE_S8:
      shift = 8 - 2 + adj;
      break;
    case MODE_S16:
      shift = 16 - 2 + adj;
      break;
    case MODE_S32:
      shift = 32 - 2;
      break;
    default:
      assert(0);  // How'd this happen?
      break;
  }
  return shift;
}

void VLMUL(xs3_vpu *vpu, const void *addr) {
  #ifdef __XS3A__
  assert_word_aligned(addr);
  #endif

  const unsigned shift = vlmul_get_shift(NN_ARCH, vpu->mode);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];
      int32_t res = ((int32_t)vpu->vR.s8[i] * (int32_t)val + (1L<<(shift - 1))) >> shift;
      vpu->vR.s8[i] = vpu_saturate_fixed(res, 8);
    }
  } else if (vpu->mode == MODE_S16 && NN_ARCH == TARGET_ARCH_VX4A) {
    // VX4 multiplies in two halves: VLMUL0 leaves the low byte's product, shifted down
    // by 8 without rounding, in vD; VLMUL1 adds the high byte's product and rounds.
    const int16_t *addr16 = (const int16_t *)addr;
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      const int32_t lo = (int32_t)(vpu->vR.u16[i] & 0xFF) * addr16[i];
      const int32_t hi = (int32_t)(int8_t)(vpu->vR.u16[i] >> 8) * addr16[i];
      vpu->vD.s16[i] = (int16_t)(lo >> 8);
      vpu->vR.s16[i] = (int16_t)vpu_saturate_fixed(round_shr(vpu->vD.s16[i] + hi, 7), 16);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int64_t val = addr16[i];
      int64_t res = ((int64_t)vpu->vR.s16[i] * (int64_t)val + (1LL<<(shift - 1))) >> shift;
      vpu->vR.s16[i] = vpu_saturate_fixed(res, 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      int64_t res = ((int64_t)vpu->vR.s32[i] * (int64_t)val + (1LL<<(shift - 1))) >> shift;
      vpu->vR.s32[i] = vpu_saturate_fixed(res, 32);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VDEPTH1(xs3_vpu *vpu) {
  uint32_t bits = 0;

  if (vpu->mode == MODE_S8) {
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      if (vpu->vR.s8[i] < 0) bits |= (1 << i);
    }
  } else if (vpu->mode == MODE_S16) {
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      if (vpu->vR.s16[i] < 0) bits |= (1 << i);
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      if (vpu->vR.s32[i] < 0) bits |= (1 << i);
    }
  } else {
    assert(0);
  }

  memset(&(vpu->vR), 0, sizeof(vpu_vector_t));
  vpu->vR.s32[0] = bits;
}

void VDEPTH8(xs3_vpu *vpu) {
  vpu_vector_t vec_tmp;
  memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
  memset(&(vpu->vR), 0, sizeof(vpu_vector_t));

  if (vpu->mode == MODE_S16) {
    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
      vpu->vR.s8[i] = vpu_saturate_fixed(elm >> 8, 8);
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t elm = ((int64_t)vec_tmp.s32[i]) + (1 << 23);
      vpu->vR.s8[i] = vpu_saturate_fixed(elm >> 24, 8);
    }
  } else {
    assert(0);
  }
}

void VDEPTH16(xs3_vpu *vpu) {
  if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t elm = ((int64_t)vpu->vR.s32[i]) + (1 << 15);
      vpu->vR.s16[i] = vpu_saturate_fixed(elm >> 16, 16);
    }

    for (int i = VPU_INT32_EPV; i < VPU_INT16_EPV; i++) {
      vpu->vR.s16[i] = 0;
    }
  } else {
    assert(0);
  }
}

static char signof(int x) { return (x >= 0 ? ' ' : '-'); }

void vpu_sim_mem_print(void *address, vector_mode mode) {
  int8_t *vC8 = (int8_t *)address;
  int16_t *vC16 = (int16_t *)address;
  int32_t *vC32 = (int32_t *)address;
  switch (mode) {
    case MODE_S8:
      printf("8-bit:\n");
      for (int i = 0; i < VPU_INT8_EPV; i++) {
        printf("%d\t%c0x%.2X(%d)\n", i, signof(vC8[i]), abs(vC8[i]), (int)vC8[i]);
      }
      break;

    case MODE_S16:
      printf("16-bit:\n");
      for (int i = 0; i < VPU_INT16_EPV; i++) {
        printf("%d\t%c0x%.4X(%d)\n", i, signof(vC16[i]), abs(vC16[i]),(int)vC16[i]);
      }
      break;

    case MODE_S16x8:
      printf("16x8-bit:\n");
      for (int i = 0; i < VPU_INT16_EPV; i++) {
        printf("%d\t%c0x%.4X(%d)\n", i, signof(vC8[2*i]), abs(vC8[2*i]),(int)vC8[2*i]);
      }
      break;

    case MODE_S32:
      printf("32-bit:\n");
      for (int i = 0; i < VPU_INT32_EPV; i++) {
        printf("%d\t%c0x%.8X(%d)\n", i, signof(vC32[i]), abs(vC32[i]), (int)vC32[i]);
      }
      break;

    default:
      printf("In the future this might print all possible interpretations...");
      break;
  }
  printf("\n");
}

void vpu_accu_print(xs3_vpu *vpu) {
  printf("Accumulators - Mode:%d\n", vpu->mode);
  if (vpu->mode == MODE_S8) {
    for (int i = 0; i < VPU_INT8_ACC_PERIOD; i++) {
      int32_t acc = GetAccumulator(vpu, i);
      printf("%d %d\n", i, (int)acc);
    }
  } else if ((vpu->mode == MODE_S16)|| (vpu->mode == MODE_S16x8)) {
    for (int i = 0; i < VPU_INT16_ACC_PERIOD; i++) {
      int32_t acc = GetAccumulator(vpu, i);
      printf("%d %d\n", i, (int)acc);
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_ACC_PERIOD; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      printf("%d %ld\n", i, (long)acc);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void vpu_sim_print(xs3_vpu *vpu) {
  int8_t *vC8 = vpu->vC.s8;
  int8_t *vR8 = vpu->vR.s8;
  int8_t *vD8 = vpu->vD.s8;

  int16_t *vC16 = vpu->vC.s16;
  int16_t *vR16 = vpu->vR.s16;
  int16_t *vD16 = vpu->vD.s16;

  int32_t *vC32 = vpu->vC.s32;
  int32_t *vR32 = vpu->vR.s32;
  int32_t *vD32 = vpu->vD.s32;
  switch (vpu->mode) {
    case MODE_S8:
      printf("8-bit:     vC     \t  vR     \t   vD\n");
      for (int i = 0; i < VPU_INT8_EPV; i++) {
        printf("%d\t%c0x%.2X(%d)\t%c0x%.2X(%d)\t%c0x%.2X(%d)\n", i,
               signof(vC8[i]), abs(vC8[i]), (int)vC8[i], signof(vR8[i]),
               abs(vR8[i]), (int)vR8[i], signof(vD8[i]), abs(vD8[i]),
               (int)vD8[i]);
      }
      break;

      case MODE_S16:
      printf("16-bit:  vC     \t    vR      \t    vD\n");
      for (int i = 0; i < VPU_INT16_EPV; i++) {
        printf("%d\t0x%.4X(%d)\t0x%.4X(%d)\t0x%.4X(%d)\n", i, abs(vC16[i]),
               (int)vC16[i], abs(vR16[i]), (int)vR16[i], abs(vD16[i]),
               (int)vD16[i]);
      }
      break;

    case MODE_S32:
      printf("32-bit:  vC     \t\t    vR      \t\t    vD\n");
      for (int i = 0; i < VPU_INT32_EPV; i++) {
        printf("%d\t%c0x%.8X(%d)\t%c0x%.8X(%d)\t%c0x%.8X(%d)\n", i,
               signof(vC32[i]), abs(vC32[i]), (int)vC32[i], signof(vR32[i]),
               abs(vR32[i]), (int)vR32[i], signof(vD32[i]), abs(vD32[i]),
               (int)vD32[i]);
      }
      break;

    default:
      printf("In the future this might print all possible interpretations...");
      break;
  }

  printf("\n");
}
