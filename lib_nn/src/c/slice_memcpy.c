#include <stdint.h>
#include <stddef.h>
#include <string.h>

void slice_memcpy_get_params(int *begin_dst, int *end_dst, int *in_offsets,
                             int *out_offsets, int *shape_dst, const int *begin,
                             const int *size, const int *shape,
                             const int dtype_size, const int rank) {

  // TFLite supports up to 5 dimensions, if the input is less we pad
  const int numPad = 5 - rank;
  for (int i = 0; i < 5; i++) {
    begin_dst[i] = i >= numPad ? begin[i - numPad] : 0;
    end_dst[i] = i >= numPad ? begin_dst[i] + size[i - numPad] : 1;
    shape_dst[i] = i >= numPad ? shape[i - numPad] : 1;
  }

  // Calculate number of input and output elements
  int num_elements_in = 1;
  int num_elements_out = 1;
  for (int i = 0; i < 5; i++) {
    num_elements_in *= shape_dst[i];
    num_elements_out *= end_dst[i] - begin_dst[i];
  }

  // Merge axes where possible in the end
  while (begin_dst[4] == 0 && end_dst[4] == shape_dst[4]) {
    int32_t last_begin = begin_dst[3] * shape_dst[4];
    int32_t last_end = end_dst[3] * shape_dst[4];
    int32_t last_dim = shape_dst[3] * shape_dst[4];
    memmove(begin_dst + 1, begin_dst, 3 * sizeof(int32_t));
    memmove(end_dst + 1, end_dst, 3 * sizeof(int32_t));
    memmove(shape_dst + 1, shape_dst, 3 * sizeof(int32_t));
    begin_dst[0] = 0;
    end_dst[0] = 1;
    shape_dst[0] = 1;
    begin_dst[4] = last_begin;
    end_dst[4] = last_end;
    shape_dst[4] = last_dim;
  }

  // Treat dtype as an extra axis that we merge with the last axis, to use
  // vpu_memcpy if possible
  shape_dst[4] *= dtype_size;
  begin_dst[4] *= dtype_size;
  end_dst[4] *= dtype_size;

  // Calculate offsets
  in_offsets[0] = num_elements_in / shape_dst[0] * dtype_size;
  out_offsets[0] = num_elements_out / (end_dst[0] - begin_dst[0]) * dtype_size;
  for (int i = 1; i < 4; i++) {
    in_offsets[i] = in_offsets[i - 1] / shape_dst[i];
    out_offsets[i] = out_offsets[i - 1] / (end_dst[i] - begin_dst[i]);
  }
}

void slice_memcpy(int8_t *dst, int8_t *src, int32_t *in_offsets,
                  int32_t *out_offsets, int32_t *begin, int32_t *end,
                  void (*memcpy_func)(void *, void *, size_t)) {
  const int memcpy_size = end[4] - begin[4];
  const int8_t *src_addr = (int8_t *)src + begin[4];
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
          memcpy_func((int8_t *)dst + out_idx3, src_addr + in_idx3,
                      memcpy_size);
        }
      }
    }
  }
}
