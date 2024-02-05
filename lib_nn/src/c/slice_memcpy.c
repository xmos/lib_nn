#include <stdint.h>

void slice_memcpy(int8_t *dst, int8_t *src, int32_t *in_offsets,
                  int32_t *out_offsets, int32_t *begin, int32_t *end,
                  void (*memcpy_func)(void *, const void *, unsigned long)) {
  for (int i0 = begin[0]; i0 < end[0]; i0++) {
    const int32_t in_idx0 = i0 * in_offsets[0];
    const int32_t out_idx0 = (i0 - begin[0]) * out_offsets[0];
    for (int i1 = begin[1]; i1 < end[1]; i1++) {
      const int32_t in_idx1 = in_idx0 + i1 * in_offsets[1];
      const int32_t out_idx1 = out_idx0 + (i1 - begin[1]) * out_offsets[1];
      for (int i2 = begin[2]; i2 < end[2]; i2++) {
        const int32_t in_idx2 = in_idx1 + i2 * in_offsets[2];
        const int32_t out_idx2 = out_idx1 + (i2 - begin[2]) * out_offsets[2];
        for (int i3 = begin[3]; i3 < end[3]; i3++) {
          const int32_t in_idx3 = in_idx2 + i3 * in_offsets[3];
          const int32_t out_idx3 = out_idx2 + (i3 - begin[3]) * out_offsets[3];
          memcpy_func((int8_t *)dst + out_idx3,
                      (int8_t *)src + in_idx3 + begin[4], out_offsets[3]);
        }
      }
    }
  }
}
