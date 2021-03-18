
#include "Rand.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include <ctime>
#include <iostream>
#include <cstring>


using namespace nn::filt2d;
using namespace nn::test;


int Rand::pseudo_rand()
{
    const int a = 1013904223;
    const int c = 1664525;
    this->state = (int)((long long)a * this->state + c);
    return this->state;
}

void Rand::pseudo_rand_bytes(void* dst, size_t size)
{
  int* dsti = static_cast<int*>(dst);

  while(size >= sizeof(int)){
    dsti[0] = pseudo_rand();
    dsti = &dsti[1];
    size -= sizeof(int);
  }

  if(size){
    int x = pseudo_rand();
    memcpy(dsti, &x, size);
  }
}


Rand::Rand()
{
  time_t t = time(nullptr);
  this->state = static_cast<int>(t);
}
