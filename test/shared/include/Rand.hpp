#pragma once

#include <cassert>
#include <iostream>

namespace nn {
  namespace test {


class Rand {

  protected:

    int state;

    int pseudo_rand();
    void pseudo_rand_bytes(void* dst, size_t size);

  public:

    Rand();

    Rand(int seed) : state(seed) {}

    void setSeed(int seed) { this->state = seed; }

    /**
     * Get a pseudo-random value of type `T`. 
     * 
     * The generic implementation uses `T`'s default constructor (which must exist), and fills 
     * it with pseudo-random bytes.
     * 
     * Some types that do not have default constructors may have specializations defined.
     */
    template <typename T>
    T rand();

    /**
     * Get  pseudo-random value of type `T` between the specified values.
     * 
     * The value `x` returned meets the following constraint:
     *     `min_inclusive <= x <= max_inclusive`
     */
    template <typename T>
    T rand(T min_inclusive, T max_inclusive);

    /**
     * Get a pseudo-random value of type `T` using the supplied parameters.
     * 
     * The idea here is that other classes may want to allow random generation of objects subject
     * to non-trivial constraints. Specializations of this function can be defined elsewhere,
     * using any appropriate parameter type `T_params`  
     */
    template <typename T, typename T_params>
    T rand(T_params params);

};



template<typename T>
T Rand::rand()
{
  if(sizeof(T) <= sizeof(int)) {
    return static_cast<T>(pseudo_rand());
  } else {
    T res;
    pseudo_rand_bytes(&res, sizeof(T));
    return res;
  }
}

template<typename T>
T Rand::rand(T min_inclusive, T max_inclusive)
{
  assert(min_inclusive <= max_inclusive);
  T span = max_inclusive - min_inclusive + 1;
  T x = static_cast<T>(((unsigned)pseudo_rand()) % span);
  return min_inclusive + x;
}

// Specializations defined in Rand.cpp need to be declared here or the compiler will choose
// the generic implementation.
// template <> unsigned Rand::rand<unsigned>();

}}