
#pragma once

#include "nn_types.h"

#include <vector>
#include <initializer_list>
#include <cassert>

namespace nn {

/**
 * TensorWrap takes a pointer to a 1-or-more-dimensional array and provides the user with a 
 * more convenient indexer syntax for accessing values.
 * 
 * Because the version of C++ used does not support multi-valued indexing, instead, an initializer
 * list is used.
 * 
 * @code
 *    int buffer[24] = { ... }; // actual dimensions are [2][3][4]
 *    auto tensor = TensorWrap<int>(buffer, {2, 3, 4});
 *    int val = tensor[{1,0,2}]; // == buffer[1*(3*4)+0*(4)+2] == buffer[14];
 * @endcode
 * 
 * Note: This class does not attempt to prevent the user from going beyond the bounds of
 *       the wrapped tensor, as that is sometimes desired.
 * 
 * Note: As dynamic memory is used, `TensorWrap` should not be used in code which is executed
 *       at inference time.
 */
template <typename T>
class TensorWrap {

  protected:

    std::vector<int> dims;
    std::vector<mem_stride_t> strides;
    T* tensor;

  public:

    TensorWrap(T* tensor, std::initializer_list<int> dims)
      : dims(dims), tensor(tensor), strides(dims.size())
    {
      int acc = 1;
      for(int d = dims.size()-1; d >= 0; --d){
        strides[d] = acc;
        acc *= this->dims[d];
      }
    }

    int GetDim(int dim_index) const { return dims[dim_index]; }

    T& operator[](std::initializer_list<int> indices);

};


template <typename T>
T& TensorWrap<T>::operator[](
    std::initializer_list<int> indices)
{
  int acc = 0;
  int d = 0;
  for(auto dex: indices)
    acc += strides[d++] * dex;

  
  return tensor[acc];
}

}