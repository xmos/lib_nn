
#include "../src/cpp/f2d_c_types.h"
#include "../src/cpp/Filter2d_util.hpp"
#include "../src/cpp/Filter2d.hpp"
#include "../src/cpp/ImageGeometry.hpp"
#include "../src/cpp/util.h"

#include <iostream>


EXTERN_C void foo(void* p);

__attribute__((section(".ExtMem_data")))
int32_t mem_block[1000] = {0};


int main()
{

  std::cout << "Before foo()!" << std::endl;

  foo((void*) &mem_block[0]);

  std::cout << "After foo()!" << std::endl;

  for(int i = 0; i < 5; i++){
    std::cout << i << ": " << mem_block[i] << std::endl;
  }

  return 0;
}