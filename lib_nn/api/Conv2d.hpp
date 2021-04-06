
#include "Filter2D.hpp"

#include <cmath>


class Conv2dVaildDirect : public Filter2D {
  public:
    Conv2dVaildDirect(AbstractKernelParams * kparams, Im_to_col_valid * memcpy_handler, 
      MatMulDirectFn * aggregate_handler, OT_int8 * ot_handler);

    Conv2dVaildDirect(ImageParams &Y, ImageRegion &Y_region,
      ImageParams &X, WindowGeometry &K, int input_ch_per_output, int8_t * weights);
};

class Conv2dVaildIndirect : public Filter2D {
  public:
    Conv2dVaildIndirect(AbstractKernelParams * kparams, MatMulFn * memcpy_handler, 
      Im_to_col_valid * aggregate_handler, OT_int8 * ot_handler);
};

class Conv2dPaddedIndirect : public Filter2D {
  public:
    Conv2dPaddedIndirect(AbstractKernelParams * kparams, MatMulFn * memcpy_handler, 
      Im_to_col_padded * aggregate_handler, OT_int8 * ot_handler);
};
