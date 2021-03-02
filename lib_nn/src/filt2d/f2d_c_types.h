#pragma once


#ifdef __cplusplus
  #include <cstdint>
extern "C" {
#else
  #include "stdint.h"
#endif


typedef int32_t mem_stride_t;
typedef uint32_t channel_count_t;

typedef struct {
  int16_t  high[16];
  uint16_t low[16];
} vpu_split_acc32_t;

typedef struct {
    uint16_t shift1[ 16 ];
    int16_t  scale[ 16 ];
    int16_t  offset_scale[ 16 ];
    int16_t  offset[ 16 ];
    uint16_t shift2[ 16 ];
} nn_acc32_to_int8_params_t;

typedef struct  {
  struct {
    int32_t const row;
    int32_t const col;
    int32_t const channel;
  } start;

  struct {
    int32_t const rows;
    int32_t const cols;
    int32_t const channels;
  } stride;
} InputCoordTransform;

typedef struct {
  int16_t top;
  int16_t left;
  int16_t bottom;
  int16_t right;
} padding_t;


#ifdef __cplusplus
} // extern "C"
#endif