// Copyright 2020 XMOS LIMITED. This Software is subject to the terms of the 
// XMOS Public License: Version 1
#pragma once

#include <stdint.h>
#include <assert.h>
#include <string.h>

#include "nn_types.h"
#include "xs3_vpu.h"

#ifdef __cplusplus
extern "C" {
#endif



typedef union {
    uint8_t  u8[VPU_INT8_EPV];
    int8_t   s8[VPU_INT8_EPV];

    uint16_t u16[VPU_INT16_EPV];
    int16_t  s16[VPU_INT16_EPV];

    uint32_t u32[VPU_INT32_EPV];
    int32_t  s32[VPU_INT32_EPV];
} vpu_vector_t;

typedef enum {
    MODE_S32 = 0x00,
    MODE_S16 = 0x100,
    MODE_S8  = 0x200,
} vector_mode;

typedef struct {
    vector_mode mode;
    vpu_vector_t vR;
    vpu_vector_t vD;
    vpu_vector_t vC;
} xs3_vpu;


void VSETC(
    xs3_vpu* vpu,
    const vector_mode mode);
void VCLRDR(xs3_vpu* vpu);
void VLDR(xs3_vpu* vpu, const void* addr);
void VLDD(xs3_vpu* vpu, const void* addr);
void VLDC(xs3_vpu* vpu, const void* addr);
void VSTR(const xs3_vpu* vpu, void* addr);
void VSTD(const xs3_vpu* vpu, void* addr);
void VSTC(const xs3_vpu* vpu, void* addr);
void VSTRPV(const xs3_vpu* vpu, void* addr, unsigned mask);
void VLMACC(xs3_vpu* vpu, const void* addr);
void VLMACCR(xs3_vpu* vpu, const void* addr);
void VLMACCR1(xs3_vpu* vpu, const void* addr);
void VLSAT(xs3_vpu* vpu, const void* addr);
void VLASHR(xs3_vpu* vpu, const void* addr, const int32_t shr);
void VLADD(xs3_vpu* vpu, const void* addr);
void VLSUB(xs3_vpu* vpu, const void* addr);
void VLMUL(xs3_vpu* vpu, const void* addr);
void VDEPTH1(xs3_vpu* vpu);
void VDEPTH8(xs3_vpu* vpu);
void VDEPTH16(xs3_vpu* vpu);

/** Print vector register contents based on current vector_mode **/
void vpu_sim_print(xs3_vpu* vpu);
void vpu_sim_mem_print(void * address, vector_mode mode);

//Function for implementing the saturation logic within the VPU.
int64_t vpu_saturate(
    const int64_t input,
    const unsigned bits);

#ifdef __cplusplus
} // extern "C"
#endif


#ifdef __cplusplus

namespace nn {

  class VPU {

    private:

      xs3_vpu vpu;

    public:

      vpu_vector_t& vD() { return this->vpu.vD;   }
      vpu_vector_t& vR() { return this->vpu.vR;   }
      vpu_vector_t& vC() { return this->vpu.vC;   }
      vector_mode mode() { return this->vpu.mode; }


      void vsetc(const vector_mode mode)         { VSETC(&this->vpu, mode);         }
      void vclrdr()                              { VCLRDR(&this->vpu);              }
      void vldr(void const* addr)                { VLDR(&this->vpu, addr);          }
      void vldd(void const* addr)                { VLDD(&this->vpu, addr);          }
      void vldc(void const* addr)                { VLDC(&this->vpu, addr);          }
      void vstr(void* addr)                      { VSTR(&this->vpu, addr);          }
      void vstd(void* addr)                      { VSTD(&this->vpu, addr);          }
      void vstc(void* addr)                      { VSTC(&this->vpu, addr);          }
      void vstrpv(void* addr, uint32_t mask)     { VSTRPV(&this->vpu, addr, mask);  }
      void vlmacc(void const* addr)              { VLMACC(&this->vpu, addr);        }
      void vlmaccr(void const* addr)             { VLMACCR(&this->vpu, addr);       }
      void vlmaccr1(void const* addr)            { VLMACCR1(&this->vpu, addr);      }
      void vlsat(void const* addr)               { VLSAT(&this->vpu, addr);         }
      void vlashr(void const* addr, int32_t shr) { VLASHR(&this->vpu, addr, shr);   }
      void vladd(void const* addr)               { VLADD(&this->vpu, addr);         }
      void vlsub(void const* addr)               { VLSUB(&this->vpu, addr);         }
      void vlmul(void const* addr)               { VLMUL(&this->vpu, addr);         }
      void vdepth1()                             { VDEPTH1(&this->vpu);             }
      void vdepth8()                             { VDEPTH8(&this->vpu);             }
      void vdepth16()                            { VDEPTH16(&this->vpu);            }

  };
}

#endif