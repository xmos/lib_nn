#pragma once

#ifdef __cplusplus
  #define EXTERN_C extern "C"
#else
  #define EXTERN_C 
#endif


#include <cstdint>

namespace nn {
namespace filt2d {


template <typename T>
static inline T* advancePointer(T* orig, int32_t offset_bytes)
{
  return (T*) (((char*)orig)+offset_bytes);
}


EXTERN_C typedef struct {
  int16_t  high[16];
  uint16_t low[16];
} vpu_split_acc32_t;




EXTERN_C typedef struct {
  int16_t top;
  int16_t left;
  int16_t bottom;
  int16_t right;

  void makeUnsigned(){
    top = (top <= 0)? 0 : top;
    left = (left <= 0)? 0 : left;
    bottom = (bottom <= 0)? 0 : bottom;
    right = (right <= 0)? 0 : right;
  }
} padding_t;



}}