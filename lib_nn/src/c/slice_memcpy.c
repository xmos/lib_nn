#include <stddef.h>
#include <stdint.h>
#include <string.h>

void slice_reshape_params(int *shape, int *begin, int *end, size_t dtype_size) {
  // Merge axes where possible in the end
  while (begin[4] == 0 && end[4] == shape[4]) {
    int32_t last_begin = begin[3] * shape[4];
    int32_t last_end = end[3] * shape[4];
    int32_t last_dim = shape[3] * shape[4];
    memmove(begin + 1, begin, 3 * sizeof(int32_t));
    memmove(end + 1, end, 3 * sizeof(int32_t));
    memmove(shape + 1, shape, 3 * sizeof(int32_t));
    begin[0] = 0;
    end[0] = 1;
    shape[0] = 1;
    begin[4] = last_begin;
    end[4] = last_end;
    shape[4] = last_dim;
  }

  // Merge axes where possible in the beginning
  int first_axis = 0;
  for (int i = 0; i < 4; i++)
    if (begin[i] == 0 && end[i] == shape[i])
      first_axis += 1;
    else
      break;

  int first_dim = 1, first_begin = 1, first_end = 1;
  for (int i = 0; i < first_axis; i++) {
    first_dim *= shape[i];
    first_begin *= begin[i];
    first_end *= end[i];
    shape[i] = 1;
    begin[i] = 0;
    end[i] = 1;
  }

  if (first_axis > 0) {
    shape[first_axis - 1] = first_dim;
    begin[first_axis - 1] = first_begin;
    end[first_axis - 1] = first_end;
  }

  // Treat dtype as an extra axis that we merge with the last axis, to use
  // vpu_memcpy if possible
  shape[4] *= dtype_size;
  begin[4] *= dtype_size;
  end[4] *= dtype_size;
}

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

  slice_reshape_params(shape_dst, begin_dst, end_dst, dtype_size);

  // Calculate offsets
  in_offsets[0] = num_elements_in / shape_dst[0] * dtype_size;
  out_offsets[0] = num_elements_out / (end_dst[0] - begin_dst[0]) * dtype_size;
  for (int i = 1; i < 4; i++) {
    in_offsets[i] = in_offsets[i - 1] / shape_dst[i];
    out_offsets[i] = out_offsets[i - 1] / (end_dst[i] - begin_dst[i]);
  }
}

void slice_memcpy_1d(int8_t *dst, int8_t *src, size_t size, int32_t offset,
                     int32_t num_copies,
                     void (*memcpy_func)(void *, void *, size_t)) {

  for (int i = 0; i < num_copies; i++) {
    const int32_t in_idx0 = i * size;
    const int32_t out_idx0 = i * offset;
    memcpy_func((int8_t *)dst + out_idx0, (int8_t *)src + in_idx0, size);
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
