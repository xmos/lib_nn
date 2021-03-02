#pragma once

// astew: Temporary. I'm not sure what's doing it, but VSCode refuses to be told that __CYGWIN__ shouldn't be defined,
//        and that's messing up intellisense.
#ifdef __CYGWIN__
  #undef __CYGWIN__
#endif

#ifdef __cplusplus
  #define EXTERN_C  extern "C"
  #include <cstdint>
#else
  #define EXTERN_C  
  #include <stdint.h>
#endif

#include "f2d_c_types.h" 

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {  
  uint32_t patch_bytes;  
  mem_stride_t K_cout_stride;
} conv2d_aggregate_deep_patch_int8_params_t; 

#ifdef __cplusplus
}
#endif

EXTERN_C void conv2d_aggregate_deep_patch_int8(
  vpu_split_acc32_t* accumulators,
  const int8_t* patch,
  const int8_t* kernel,
  const conv2d_aggregate_deep_patch_int8_params_t* params,
  const channel_count_t out_chans);


EXTERN_C void conv2d_output_transform_symmetric_int8(
      int8_t * output,
      const vpu_split_acc32_t* accumulator,
      const nn_acc32_to_int8_params_t* params,
      unsigned const channels_out);

EXTERN_C void conv2d_output_transform_asymmetric_int8(
      int8_t * output,
      const vpu_split_acc32_t* accumulator,
      const nn_acc32_to_int8_params_t* params,
      unsigned const channels_out);



