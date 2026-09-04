// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>

#ifdef NN_USE_REF
void lookup8_ref(
  uint8_t *Y,
  const uint8_t *X,
  const uint8_t *lut,
  const unsigned elm_start,
  const unsigned elm_count
) {
  for (unsigned i = elm_start; i < elm_start + elm_count; i++) {
    Y[i] = lut[X[i]];
  }
}

#else

extern void lookup8_asm(
  uint8_t *Y,
  const uint8_t *X,
  const uint8_t *lut,
  const unsigned elm_start,
  const unsigned elm_count
);

#endif

void lookup8(uint8_t *Y, const uint8_t *X, const uint8_t *lut,
             const unsigned elm_start, const unsigned elm_count) {
#ifdef NN_USE_REF
  lookup8_ref(Y, X, lut, elm_start, elm_count);
#else
  lookup8_asm(Y, X, lut, elm_start, elm_count);
#endif // NN_USE_REF
}
