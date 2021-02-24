#pragma once


#ifdef __cplusplus
  #include <cstdint>
extern "C" {
#else
  #include "stdint.h"
#endif


typedef int32_t mem_stride_t;


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
  int16_t row_bytes;
  int16_t col_bytes;
  int16_t chan_bytes;
  int16_t const zero = 0;
} PointerCovector;

typedef struct {
  struct {
    int16_t const top;
    int16_t const left;
    int16_t const bottom;
    int16_t const right;
  } initial;
  struct {
    int16_t const top;
    int16_t const left;
    int16_t const bottom;
    int16_t const right;
  } stride;
} PaddingTransform;

typedef struct {
  int16_t top;
  int16_t left;
  int16_t bottom;
  int16_t right;
} padding_t;


#ifdef __cplusplus
} // extern "C"
#endif