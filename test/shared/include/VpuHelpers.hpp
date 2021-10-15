#pragma once

#include <cstdint>

namespace nn {
namespace test {

static inline void split_acc32(int16_t& acc_hi, int16_t& acc_lo,
                               const int32_t acc32) {
  acc_hi = int16_t((acc32 & 0xFFFF0000) >> 16);
  acc_lo = int16_t(acc32 & 0x0000FFFF);
}

static inline int32_t merge_acc32(const int16_t acc_hi, const int16_t acc_lo) {
  int32_t acc32 = int32_t(acc_hi) << 16;
  return acc32 &
         int32_t(uint16_t(acc_lo));  // don't let it sign-extend acc_lo. It
                                     // should be treated as unsigned!
}

template <typename T_out, typename T_in>
static inline T_out vpu_sat(T_in input, bool symmetric = true) {
  assert(sizeof(T_in) > sizeof(T_out));

  const T_out upper_bound = std::numeric_limits<T_out>::max();
  const T_out lower_bound =
      symmetric ? -upper_bound : std::numeric_limits<T_out>::min();

  input = std::min<T_in>(upper_bound, input);
  input = std::max<T_in>(lower_bound, input);
  return T_out(input);
}

}  // namespace test
}  // namespace nn