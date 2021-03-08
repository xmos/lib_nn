#include <cstdint>

#ifndef VPU_HPP_
#define VPU_HPP_

struct vpu_ring_buffer_t {
  int32_t vR[8]; 
  int32_t vD[8]; 
};

#endif //VPU_HPP_

