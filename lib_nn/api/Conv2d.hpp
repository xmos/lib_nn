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
                      MatMulInt8 *aggregator, OT_int8 *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

class Conv2dPaddedInDirect : public Filter2D {
 public:
  // Padded
  // Arbitrary input + output channel count
  Conv2dPaddedInDirect(AbstractKernel::Params *akp, ImToColPadded *memcpy,
                       MatMulInt8 *aggregator, OT_int8 *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

class Conv2dDepthwiseValidDirect : public Filter2D_DW {
 public:
  // Valid only
  // Multiple of 4 input and 4 output channels
  Conv2dDepthwiseValidDirect(AbstractKernel::Params *akp, DerefInputFn *memcpy,
                             MatMulDirectFn_DW *aggregator, OT_int8 *ot)
      : Filter2D_DW(akp, memcpy, aggregator, ot) {}
};

class Conv2dDepthwisePaddedIndirect : public Filter2D_DW {
 public:
  // Valid only
  // Multiple of 4 input and 4 output channels
  Conv2dDepthwisePaddedIndirect(AbstractKernel::Params *akp,
                                ImToColPadded *memcpy,
                                MatMulDirectFn_DW *aggregator, OT_int8 *ot)
      : Filter2D_DW(akp, memcpy, aggregator, ot) {}
};

class BNNConv2dValidDirectBinary : public Filter2D {
 public:
  // Valid only
  // Multiple of 256 input and 16 output channels
  BNNConv2dValidDirectBinary(AbstractKernel::Params *akp, DerefInputFn *memcpy,
                             MatMulBinaryDirectFn *aggregator, OT_binary *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

class BNNConv2dValidDirectInt8 : public Filter2D {
 public:
  // Valid only
  // Multiple of 256 input and 16 output channels
  BNNConv2dValidDirectInt8(AbstractKernel::Params *akp, DerefInputFn *memcpy,
                           MatMulBinaryDirectFn *aggregator,
                           OT_int8_clamped *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

class BNNConv2dValidIndirectBinary : public Filter2D {
 public:
  // Valid only
  // Multiple of 32 input and 16 output channels
  BNNConv2dValidIndirectBinary(AbstractKernel::Params *akp,
                               ImToColValid *memcpy, MatMulBinary *aggregator,
                               OT_binary *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

class BNNConv2dValidIndirectInt8 : public Filter2D {
 public:
  // Valid only
  // Multiple of 32 input and 16 output channels
  BNNConv2dValidIndirectInt8(AbstractKernel::Params *akp, ImToColValid *memcpy,
                             MatMulBinary *aggregator, OT_int8_clamped *ot)
      : Filter2D(akp, memcpy, aggregator, ot) {}
};

}  // namespace nn