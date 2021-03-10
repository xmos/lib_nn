#include <cstdint>

#ifndef VPU_HPP_
#define VPU_HPP_

struct vpu_ring_buffer_t {
  int16_t vR[16]; 
  int16_t vD[16]; 
};

#endif //VPU_HPP_

