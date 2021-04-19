
#include "Filter2D.hpp"

#include <cmath>

namespace nn {

  /**
   * MaxPool2d operator that should work for (almost) any geometry.
   * 
   * Where supported, MaxPool2d_Valid should generally be used. In particular, this operator
   * supports geometries that involve input padding or dilations other than 1.
   * 
   * @see MaxPool2d_Valid
   **/
  class MaxPool2d_Generic : public Filter2D_DW {

    public:

      static constexpr int ChannelsPerOutputGroup = VPU_INT8_EPV;


      struct Params {
        AbstractKernel::Params ak_params;
        ImToColPadded::Params  mem_params;
        MaxPoolPatchFn::Params agg_params;
        DirectWriteOutputTransform::Params ot_params;
      };

    public:

      MaxPool2d_Generic( Params* params,
                         ImToColPadded* memcopy,
                         MaxPoolPatchFn* agg,
                         DirectWriteOutputTransform* ot,
                         int8_t* scratch_mem);
  };

  /**
   * MaxPool2d operator that works for geometries which involve no input padding and 
   * where both vertical and horizontal dilation are 1.
   **/
  class MaxPool2d_Valid : public Filter2D_DW {

    public:

      static constexpr int ChannelsPerOutputGroup = VPU_INT8_EPV;

      struct Params {
        AbstractKernel::Params ak_params;
        DerefInputFn::Params  mem_params;
        MaxPoolDirectValidFn::Params agg_params;
        DirectWriteOutputTransform::Params ot_params;
      };

    public:

      static MaxPool2d_Valid Make(const Params* params,
                                  int8_t* scratch_mem);

      MaxPool2d_Valid(const Params* params,
                      int8_t* scratch_mem);
  };

}
