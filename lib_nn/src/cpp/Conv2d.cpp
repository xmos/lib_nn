
#include "Conv2d.hpp"

#include <cmath>

Conv2dVaildDirect::Conv2dVaildDirect(AbstractKernelParams * kparams, Im_to_col_valid * memcpy_handler, 
  MatMulDirectFn * aggregate_handler, OT_int8 * ot_handler):Filter2D(kparams,memcpy_handler,
  aggregate_handler, ot_handler){}

