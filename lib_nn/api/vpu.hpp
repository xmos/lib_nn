#include <cstdint>
#include <ostream>
#include <cstring>

#ifndef VPU_HPP_
#define VPU_HPP_

#include "xs3_vpu.h"

struct vpu_ring_buffer_t
{
  int16_t vR[VPU_INT16_EPV];
  int16_t vD[VPU_INT16_EPV];

#ifdef __cplusplus

  /// Get the 32-bit accumulator with the specified index
  int32_t get_acc(const int index) const
  {
    return (int32_t(vD[index]) << 16) | uint32_t(uint16_t(vR[index]));
  }

  /// Set the 32-bit accumulator with the specified index
  void set_acc(const int index, const int32_t acc)
  {
    vD[index] = int16_t(uint32_t(acc) >> 16);
    vR[index] = int16_t(acc & 0x0000FFFF);
  }

  /// Add a 32-bit value to the 32-bit accumulator with the specified index
  /// The new value is returned.
  int32_t add_acc(const int index, const int32_t add_value)
  {
    int32_t v = get_acc(index) + add_value;
    set_acc(index, v);
    return v;
  }

  /// Test equality between two vpu_ring_buffer_t
  bool operator==(vpu_ring_buffer_t other) const
  {
    return std::memcmp(this, &other, sizeof(vpu_ring_buffer_t)) == 0;
  }
#endif //__cplusplus
};

#ifdef __cplusplus
inline std::ostream &operator<<(std::ostream &stream, const vpu_ring_buffer_t &buff)
{
  stream << "[";
  for (int k = 0; k < VPU_INT16_ACC_PERIOD; ++k)
    stream << std::hex << "0x" << uint32_t(buff.get_acc(k)) << std::dec << ", ";
  stream << "]";
  return stream;
}
#endif

#endif //VPU_HPP_
