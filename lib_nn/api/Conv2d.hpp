#include <cmath>
#include <tuple>

#include "Filter2D.hpp"
#include "geom/Filter2dGeometry.hpp"

namespace nn {

class Conv2dValidDirect : public Filter2D {
 public:
  // Valid only
  // Multiple of 32 input and 16 output channels
  Conv2dValidDirect(AbstractKernel::Params *akp, DerefInputFn *memcpy,
                    MatMulDirectFn *aggregator, OT_int8 *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

class Conv2dValidIndirect : public Filter2D {
 public:
  // Valid only
  // Arbitrary input + output channel count
  Conv2dValidIndirect(AbstractKernel::Params *akp, ImToColValid *memcpy,
                      MatMulInt8 *aggregator, OT_int8 *ot, int8_t *scratch)
      : Filter2D(akp, memcpy, aggregator, ot, scratch) {}
};

class Conv2dPaddedInDirect : public Filter2D {
 public:
  // Padded
  // Arbitrary input + output channel count
  Conv2dPaddedInDirect(AbstractKernel::Params *akp, ImToColPadded *memcpy,
                       MatMulInt8 *aggregator, OT_int8 *ot, int8_t *scratch)
      : Filter2D(akp, memcpy, aggregator, ot, scratch) {}
};

class Conv2dDepthwiseValidDirect : public Filter2D_DW {
 public:
  // Valid only
  // Multiple of 32 input and 16 output channels
  Conv2dDepthwiseValidDirect(AbstractKernel::Params *akp, DerefInputFn *memcpy,
                             MatMulDirectFn *aggregator, OT_int8 *ot)
      : Filter2D_DW(akp, memcpy, aggregator, ot) {}
};

class Conv2dDepthwiseValidIndirect : public Filter2D_DW {
 public:
  // Valid only
  // Arbitrary input + output channel count
  Conv2dDepthwiseValidIndirect(AbstractKernel::Params *akp,
                               ImToColValid *memcpy, MatMulInt8 *aggregator,
                               OT_int8 *ot, int8_t *scratch)
      : Filter2D_DW(akp, memcpy, aggregator, ot, scratch) {}
};

class Conv2dDepthwisePaddedInDirect : public Filter2D_DW {
 public:
  // Padded
  // Arbitrary input + output channel count
  Conv2dDepthwisePaddedInDirect(AbstractKernel::Params *akp,
                                ImToColPadded *memcpy, MatMulInt8 *aggregator,
                                OT_int8 *ot, int8_t *scratch)
      : Filter2D_DW(akp, memcpy, aggregator, ot, scratch) {}
};
}  // namespace nn