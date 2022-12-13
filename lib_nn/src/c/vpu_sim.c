// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

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
 * vpu_saturate to the relevant bounds. Fixed 8-bit saturation.
 */
int64_t vpu_saturate_fixed(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  int64_t min_val = -max_val;
  if(bits == 8){
    min_val = INT8_MIN;
  }

  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

/**
 * Get the accumulator for the VPU's current mode
 */
static int64_t GetAccumulator(const xs3_vpu *vpu, unsigned index) {
  if (vpu->mode == MODE_S8 || vpu->mode == MODE_S16) {
    union {
      int16_t s16[2];
      int32_t s32;
    } acc;
    acc.s16[1] = vpu->vD.s16[index];
    acc.s16[0] = vpu->vR.s16[index];

    return acc.s32;
  } else {
    assert(0);  // TODO
  }
}

/**
 * Set the accumulator for the VPU's current mode
 */
static void SetAccumulator(xs3_vpu *vpu, unsigned index, int64_t acc) {
  if (vpu->mode == MODE_S8 || vpu->mode == MODE_S16) {
    unsigned mask = (1 << VPU_INT8_ACC_VR_BITS) - 1;
    vpu->vR.s16[index] = (int16_t)((unsigned)acc & mask);
    mask = mask << VPU_INT8_ACC_VR_BITS;
    vpu->vD.s16[index] =
        (int16_t)(((unsigned)acc & mask) >> VPU_INT8_ACC_VR_BITS);
  } else {
    assert(0);  // TODO
  }
}

/**
 * Rotate the accumulators following a VLMACCR
 */
static void rotate_accumulators(xs3_vpu *vpu) {
  if (vpu->mode == MODE_S8 || vpu->mode == MODE_S16) {
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
  assert_word_aligned(addr);
  memcpy(&vpu->vR.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDD(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  memcpy(&vpu->vD.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VLDC(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  memcpy(&vpu->vC.u8[0], addr, XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTR(const xs3_vpu *vpu, void *addr) {
  assert_word_aligned(addr);
  memcpy(addr, &vpu->vR.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTD(const xs3_vpu *vpu, void *addr) {
  assert_word_aligned(addr);
  memcpy(addr, &vpu->vD.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTC(const xs3_vpu *vpu, void *addr) {
  assert_word_aligned(addr);
  memcpy(addr, &vpu->vC.u8[0], XS3_VPU_VREG_WIDTH_BYTES);
}

void VSTRPV(const xs3_vpu *vpu, void *addr, unsigned mask) {
  assert_word_aligned(addr);
  int8_t *addr8 = (int8_t *)addr;

  for (int i = 0; i < 32; i++) {
    if (mask & (1UL << i)) {
      addr8[i] = vpu->vR.s8[i];
    }
  }
}

void VLMACC(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;

    for (int i = 0; i < VPU_INT8_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + (((int32_t)vpu->vC.s8[i]) * addr8[i]);

      SetAccumulator(vpu, i, vpu_saturate(acc, 32));
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + (((int32_t)vpu->vC.s16[i]) * addr16[i]);

      SetAccumulator(vpu, i, vpu_saturate(acc, 32));
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_VLMACC_ELMS; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      acc = acc + (((int64_t)vpu->vC.s32[i]) * addr32[i]);

      SetAccumulator(vpu, i, vpu_saturate(acc, 40));
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLMACCR(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    int64_t acc = GetAccumulator(vpu, VPU_INT8_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT8_EPV; i++)
      acc = acc + (((int32_t)vpu->vC.s8[i]) * addr8[i]);

    acc = vpu_saturate(acc, 32);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;
    int64_t acc = GetAccumulator(vpu, VPU_INT16_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT16_EPV; i++)
      acc = acc + (((int32_t)vpu->vC.s16[i]) * addr16[i]);

    acc = vpu_saturate(acc, 32);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;
    int32_t acc = GetAccumulator(vpu, VPU_INT32_ACC_PERIOD - 1);

    for (int i = 0; i < VPU_INT32_EPV; i++)
      acc = acc + (((int32_t)vpu->vC.s32[i]) * addr32[i]);

    acc = vpu_saturate(acc, 40);
    rotate_accumulators(vpu);
    SetAccumulator(vpu, 0, acc);
  } else {
    assert(0);  // How'd this happen?
  }
}

void VPOS(xs3_vpu *vpu) {
  if (vpu->mode == MODE_S8) {
    for (int i = 0; i < VPU_INT8_ACC_PERIOD; i++) {
      int8_t acc = (int8_t)GetAccumulator(vpu, i);
      if (acc < 0) acc = 0;
      vpu->vR.s8[i] = acc;
    }
  } else if (vpu->mode == MODE_S16) {
    for (int i = 0; i < VPU_INT16_ACC_PERIOD; i++) {
      int16_t acc = (int16_t)GetAccumulator(vpu, i);
      if (acc < 0) acc = 0;
      vpu->vR.s16[i] = acc;
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_ACC_PERIOD; i++) {
      int32_t acc = (int32_t)GetAccumulator(vpu, i);
      if (acc < 0) acc = 0;
      vpu->vR.s32[i] = acc;
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLMACCR1(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  const int32_t *addr32 = (const int32_t *)addr;
  int64_t acc = GetAccumulator(vpu, VPU_BIN_ACC_PERIOD - 1);

  for (int i = 0; i < VPU_INT32_EPV; i++) {
    int v = (((int32_t)vpu->vC.s32[i]) ^ addr32[i]);
    acc += (2 * __builtin_popcount(~v) - 32) / 2;
  }

  acc = vpu_saturate(acc, 32);
  rotate_accumulators(vpu);
  SetAccumulator(vpu, 0, acc);
}

void _VLSAT_IMPL(xs3_vpu *vpu, const void *addr, bool fixed_saturation) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const uint16_t *addr16 = (const uint16_t *)addr;

    for (int i = 0; i < VPU_INT8_ACC_PERIOD; i++) {
      int32_t acc = GetAccumulator(vpu, i);

      if (addr16[i] != 0) acc = acc + (1 << (addr16[i] - 1));  // Round
      acc = acc >> addr16[i];                                  // Shift
      int8_t val;
      if(fixed_saturation){
        val = vpu_saturate_fixed(acc, 8);                 // vpu_saturate
      } else {
        val = vpu_saturate(acc, 8);                       // vpu_saturate
      }

      vpu->vR.s8[i] = val;
    }
    memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
    memset(&vpu->vR.u8[VPU_INT8_ACC_PERIOD], 0, VPU_INT8_ACC_PERIOD);
  } else if (vpu->mode == MODE_S16) {
    const uint16_t *addr16 = (const uint16_t *)addr;

    for (int i = 0; i < VPU_INT16_ACC_PERIOD; i++) {
      int32_t acc = GetAccumulator(vpu, i);
      if (addr16[i] != 0)
        acc = acc + (1 << ((int16_t)(addr16[i] - 1)));  // Round

      acc = acc >> addr16[i];               // Shift
      int16_t val = vpu_saturate(acc, 16);  // vpu_saturate

      vpu->vR.s16[i] = val;
    }
    memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
  } else if (vpu->mode == MODE_S32) {
    const uint32_t *addr32 = (const uint32_t *)addr;

    for (int i = 0; i < VPU_INT32_ACC_PERIOD; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      if (addr32[i] != 0) acc = acc + (1 << (addr32[i] - 1));  // Round
      acc = acc >> addr32[i];                                  // Shift
      int32_t val = vpu_saturate(acc, 32);                     // vpu_saturate

      vpu->vR.s32[i] = val;
    }
    memset(&vpu->vD.u8[0], 0, XS3_VPU_VREG_WIDTH_BYTES);
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLSAT(xs3_vpu *vpu, const void *addr) {
  _VLSAT_IMPL(vpu, addr, /*fixed_saturation=*/false);
}

void VLSAT_FIXED(xs3_vpu *vpu, const void *addr) {
  _VLSAT_IMPL(vpu, addr, /*fixed_saturation=*/true);
}

void VLASHR(xs3_vpu *vpu, const void *addr, const int32_t shr) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;

    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];

      if (shr >= 7)
        val = (val < 0) ? -1 : 0;
      else if (shr > 0)
        val = (val + (1<<(shr-1))) >> shr;
      else
        val = (unsigned)val << (-shr);

      vpu->vR.s8[i] = vpu_saturate(val, 8);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int32_t val = addr16[i];
      if (shr >= 15)
        val = (val < 0) ? -1 : 0;
      else if (shr > 0)
        val = (val + (1<<(shr-1))) >> shr;
      else
        val = (int32_t)((uint64_t)(uint32_t)val << (-shr));
      vpu->vR.s16[i] = vpu_saturate(val, 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      if (shr >= 31)
        val = (val < 0) ? -1 : 0;
      else if (shr > 0)
        val = (val + (1<<(shr-1))) >> shr;
      else
        val = (unsigned)val << (-shr);
      vpu->vR.s32[i] = vpu_saturate(val, 32);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLADD(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];
      vpu->vR.s8[i] = vpu_saturate((int32_t)vpu->vR.s8[i] + val, 8);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int32_t val = addr16[i];
      vpu->vR.s16[i] = vpu_saturate((int32_t)vpu->vR.s16[i] + val, 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      vpu->vR.s32[i] = vpu_saturate((int32_t)vpu->vR.s32[i] + val, 32);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}

void VLSUB(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];
      vpu->vR.s8[i] = vpu_saturate(val - (int32_t)vpu->vR.s8[i], 8);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int32_t val = addr16[i];
      vpu->vR.s16[i] = vpu_saturate(val - (int32_t)vpu->vR.s16[i], 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      vpu->vR.s32[i] = vpu_saturate(val - (int32_t)vpu->vR.s32[i], 32);
    }
  } else {
    assert(0);  // How'd this happen?
  }
}
void VLMUL(xs3_vpu *vpu, const void *addr) {
  assert_word_aligned(addr);
  if (vpu->mode == MODE_S8) {
    const int8_t *addr8 = (const int8_t *)addr;
    for (int i = 0; i < VPU_INT8_EPV; i++) {
      int32_t val = addr8[i];
      int32_t res = ((int32_t)vpu->vR.s8[i] * val + (1<<5)) >> 6;  // TODO use macros
      vpu->vR.s8[i] = vpu_saturate(res, 8);
    }
  } else if (vpu->mode == MODE_S16) {
    const int16_t *addr16 = (const int16_t *)addr;

    for (int i = 0; i < VPU_INT16_EPV; i++) {
      int64_t val = addr16[i];
      int64_t res =
          ((int64_t)vpu->vR.s16[i] * val ) >> 14;  // TODO use macros
      vpu->vR.s16[i] = vpu_saturate(res, 16);
    }
  } else if (vpu->mode == MODE_S32) {
    const int32_t *addr32 = (const int32_t *)addr;

    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t val = addr32[i];
      int64_t res = (vpu->vR.s32[i] * val + (1<<29)) >> 30;  // TODO use macros
      vpu->vR.s32[i] = vpu_saturate(res, 32);
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
      vpu->vR.s8[i] = vpu_saturate(elm >> 8, 8);
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t elm = ((int64_t)vec_tmp.s32[i]) + (1 << 23);
      vpu->vR.s8[i] = vpu_saturate(elm >> 24, 8);
    }
  } else {
    assert(0);
  }
}

void VDEPTH16(xs3_vpu *vpu) {
  if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_EPV; i++) {
      int64_t elm = ((int64_t)vpu->vR.s32[i]) + (1 << 15);
      vpu->vR.s16[i] = vpu_saturate(elm >> 16, 16);
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
        printf("%d\t%c0x%0.2X(%d)\n", i, signof(vC8[i]), abs(vC8[i]),
               (int)vC8[i]);
      }
      break;

    case MODE_S16:
      printf("16-bit:\n");
      for (int i = 0; i < VPU_INT16_EPV; i++) {
        printf("%d\t%c0x%0.4X(%d)\n", i, signof(vC16[i]), abs(vC16[i]),
               (int)vC16[i]);
      }
      break;

    case MODE_S32:
      printf("32-bit:\n");
      for (int i = 0; i < VPU_INT32_EPV; i++) {
        printf("%d\t%c0x%0.8X(%d)\n", i, signof(vC32[i]), abs(vC32[i]),
               (int)vC32[i]);
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
      printf("%d %d\n", i, acc);
    }
  } else if (vpu->mode == MODE_S16) {
    for (int i = 0; i < VPU_INT16_ACC_PERIOD; i++) {
      int32_t acc = GetAccumulator(vpu, i);
      printf("%d %d\n", i, acc);
    }
  } else if (vpu->mode == MODE_S32) {
    for (int i = 0; i < VPU_INT32_ACC_PERIOD; i++) {
      int64_t acc = GetAccumulator(vpu, i);
      printf("%d %lld\n", i, acc);
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
        printf("%d\t%c0x%0.2X(%d)\t%c0x%0.2X(%d)\t%c0x%0.2X(%d)\n", i,
               signof(vC8[i]), abs(vC8[i]), (int)vC8[i], signof(vR8[i]),
               abs(vR8[i]), (int)vR8[i], signof(vD8[i]), abs(vD8[i]),
               (int)vD8[i]);
      }
      break;

    case MODE_S16:
      printf("16-bit:  vC     \t    vR      \t    vD\n");
      for (int i = 0; i < VPU_INT16_EPV; i++) {
#if 0

        printf("%d\t%c0x%0.4X(%d)\t%c0x%0.4X(%d)\t%c0x%0.4X(%d)\n", i,
               signof(vC16[i]), abs(vC16[i]), (int)vC16[i], signof(vR16[i]),
               abs(vR16[i]), (int)vR16[i], signof(vD16[i]), abs(vD16[i]),
               (int)vD16[i]);
#else
        printf("%d\t0x%0.4X(%d)\t0x%0.4X(%d)\t0x%0.4X(%d)\n", i, abs(vC16[i]),
               (int)vC16[i], abs(vR16[i]), (int)vR16[i], abs(vD16[i]),
               (int)vD16[i]);
#endif
      }
      break;

    case MODE_S32:
      printf("32-bit:  vC     \t\t    vR      \t\t    vD\n");
      for (int i = 0; i < VPU_INT32_EPV; i++) {
        printf("%d\t%c0x%0.8X(%d)\t%c0x%0.8X(%d)\t%c0x%0.8X(%d)\n", i,
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
