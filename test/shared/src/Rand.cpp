
#include "Rand.hpp"

#include <cmath>
#include <cstring>
#include <ctime>
#include <iostream>
#include <tuple>

using namespace nn::test;

int Rand::pseudo_rand() {
  const int a = 1664525;
  const int c = 1013904223;
  this->state = (int)((long long)a * this->state + c);
  return this->state;
}

void Rand::pseudo_rand_bytes(void* dst, size_t size) {
  int* dsti = static_cast<int*>(dst);

  while (size >= sizeof(int)) {
    dsti[0] = pseudo_rand();
    dsti = &dsti[1];
    size -= sizeof(int);
  }

  if (size) {
    int x = pseudo_rand();
    memcpy(dsti, &x, size);
  }
}

Rand::Rand() {
  time_t t = time(nullptr);
  this->state = static_cast<int>(t);
}

void Rand::rand_bytes(void* dst, size_t size) {
  this->pseudo_rand_bytes(dst, size);
}

template <>
int8_t Rand::get_rand<int8_t>(Tag<int8_t>) {
  return int8_t(pseudo_rand());
}

/**
 * 50/50 true/false
 */
template <>
bool Rand::get_rand<bool>(Tag<bool>) {
  return this->rand<int>(0, 1);
}

/**
 * Random float with uniform distribution over [-1.0f, 1.0f)
 */
template <>
float Rand::get_rand<float>(Tag<float>) {
  auto t = this->rand<int32_t>();
  return t * ldexpf(1, -31);
}

/**
 * Random double with uniform distribution over [-1.0f, 1.0f)
 */
template <>
double Rand::get_rand<double>(Tag<double>) {
  auto t = this->rand<int64_t>();
  return t * ldexp(1, -63);
}

/**
 * Random float with uniform distribution over [min, max)
 */
template <>
float Rand::get_rand<float>(Tag<float>, float min, float max) {
  auto t = this->rand<uint32_t>();
  return (t * ldexpf(1, -32)) * (max - min) + min;
}

/**
 * Random double with uniform distribution over [min, max)
 */
template <>
double Rand::get_rand<double>(Tag<double>, double min, double max) {
  auto t = this->rand<uint64_t>();
  return (t * ldexpf(1, -64)) * (max - min) + min;
}