#include <cstdint>
#include <cstring>
#include <ostream>

#ifndef LIB_NN_VPU_HPP_
#define LIB_NN_VPU_HPP_

#include "xs3_vpu.h"

struct VPURingBuffer {
  int16_t vR[VPU_INT16_EPV];
  int16_t vD[VPU_INT16_EPV];

  /// Get the 32-bit accumulator with the specified index
  int32_t GetAccu(const int index) const {
    return (int32_t(vD[index]) << 16) | uint32_t(uint16_t(vR[index]));
  }

  /// Set the 32-bit accumulator with the specified index
  void SetAccu(const int index, const int32_t acc) {
    vD[index] = int16_t(uint32_t(acc) >> 16);
    vR[index] = int16_t(acc & 0x0000FFFF);
  }

  /// Add a 32-bit value to the 32-bit accumulator with the specified index
  /// The new value is returned.
  int32_t AddAccu(const int index, const int32_t add_value) {
    int32_t v = GetAccu(index) + add_value;
    SetAccu(index, v);
    return v;
  }

  /// Test equality between two VPURingBuffer
  bool operator==(VPURingBuffer other) const {
    return std::memcmp(this, &other, sizeof(VPURingBuffer)) == 0;
  }
};

inline std::ostream &operator<<(std::ostream &stream,
                                const VPURingBuffer &buff) {
  stream << "[";
  for (int k = 0; k < VPU_INT16_ACC_PERIOD; ++k)
    stream << std::hex << "0x" << uint32_t(buff.GetAccu(k)) << std::dec << ", ";
  stream << "]";
  return stream;
}

#endif  // LIB_NN_VPU_HPP_
