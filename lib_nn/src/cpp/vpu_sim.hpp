

#include <cstdint>

#include "../c/vpu_sim.h"

namespace nn {

  class VPU {

    private:

      xs3_vpu vpu;

    public:

      vpu_vector_t& vD() { return this->vpu.vD;   }
      vpu_vector_t& vR() { return this->vpu.vR;   }
      vpu_vector_t& vC() { return this->vpu.vC;   }
      vector_mode mode() { return this->vpu.mode; }


      void vsetc(const vector_mode mode) {  VSETC(&this->vpu, mode);  }
      void vclrdr() { VCLRDR(&this->vpu);   }
      void vldr(void const* addr) { VLDR(&this->vpu, addr);  }
      void vldd(void const* addr) { VLDD(&this->vpu, addr);  }
      void vldc(void const* addr) { VLDC(&this->vpu, addr);  }
      void vstr(void* addr) { VSTR(&this->vpu, addr);  }
      void vstd(void* addr) { VSTD(&this->vpu, addr);  }
      void vstc(void* addr) { VSTC(&this->vpu, addr);  }
      void vstrpv(void* addr, uint32_t mask) { VSTRPV(&this->vpu, addr, mask);  }
      void vlmacc(void const* addr) { VLMACC(&this->vpu, addr);  }
      void vlmaccr(void const* addr) { VLMACCR(&this->vpu, addr);  }
      void vlmaccr1(void const* addr) { VLMACCR1(&this->vpu, addr);  }
      void vlsat(void const* addr) { VLSAT(&this->vpu, addr);  }
      void vlashr(void const* addr, int32_t shr) { VLASHR(&this->vpu, addr, shr);  }
      void vladd(void const* addr) { VLADD(&this->vpu, addr);  }
      void vlsub(void const* addr) { VLSUB(&this->vpu, addr);  }
      void vlmul(void const* addr) { VLMUL(&this->vpu, addr);  }
      void vdepth1() { VDEPTH1(&this->vpu);  }
      void vdepth8() { VDEPTH8(&this->vpu);  }
      void vdepth16() { VDEPTH16(&this->vpu);  }

  };
}