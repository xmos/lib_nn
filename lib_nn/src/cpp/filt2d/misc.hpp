#pragma once

#include "nn_api.h"


#include <cstdint>
#include <iostream>
#include <algorithm>

namespace nn {

template <typename T>
static inline T* advancePointer(T* orig, int32_t offset_bytes)
{
  return (T*) (((char*)orig)+offset_bytes);
}


C_API typedef struct {
  int16_t  high[16];
  uint16_t low[16];
} vpu_split_acc32_t;




C_API typedef struct {
  int16_t top;
  int16_t left;
  int16_t bottom;
  int16_t right;

  void MakeUnsigned(){
    top = std::max<int16_t>(0, top);
    left = std::max<int16_t>(0, left);
    bottom = std::max<int16_t>(0, bottom);
    right = std::max<int16_t>(0, right);
  }

  bool HasPadding() const {
    return top > 0 || left > 0 || bottom > 0 || right > 0;
  }
} padding_t;

inline std::ostream& operator<<(std::ostream &stream, const padding_t &pad){
  return stream << "(" << pad.top << "," << pad.left << "," << pad.bottom << "," << pad.right << ")";
}

}