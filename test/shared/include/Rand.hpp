#pragma once

#include <cassert>
#include <iostream>
#include <tuple>
#include <vector>

#include "template_magic.hpp"

namespace nn {
namespace test {

/// @TODO: Add a "choose()" (or "pick()"?) method to Rand, which randomly
/// chooses one of
///        a list of options.

class Rand {
 protected:
  int state;

  // Because functions can't be overloaded on return type only, to overload
  // get_rand() we need to actually pass it a type indicating what we want.
  template <typename T>
  struct Tag {};

  // Default implementation
  template <typename T>
  T get_rand(Tag<T>);

  // Because we can't just statically cast a std::tuple<> into existence, this
  // overload captures them.
  template <typename... Ts>
  std::tuple<Ts...> get_rand(Tag<std::tuple<Ts...> >);

  template <typename T, size_t N_size>
  std::array<T, N_size> get_rand(Tag<std::array<T, N_size> >);

  template <typename T>
  T get_rand(Tag<T>, T min, T max);

  template <int... indices, typename... Ts>
  std::tuple<Ts...> get_rand(Tag<std::tuple<Ts...> >, index_list<indices...>,
                             std::tuple<Ts...> min, std::tuple<Ts...> max);

  template <typename... Ts>
  std::tuple<Ts...> get_rand(Tag<std::tuple<Ts...> >, std::tuple<Ts...> min,
                             std::tuple<Ts...> max);

  template <typename... Ts>
  std::tuple<Ts...> get_rand_tuple(Tag<std::tuple<Ts...> >,
                                   std::tuple<Ts, Ts>... min_max_inclusive);

 protected:
  int pseudo_rand();
  void pseudo_rand_bytes(void *dst, size_t size);

 public:
  Rand();

  constexpr Rand(int seed) noexcept : state(seed) {}

  void setSeed(int seed) { this->state = seed; }

  int getSeed() { return this->state; }

  /**
   * Get a pseudo-random value of type `T`.
   *
   * The generic implementation uses `T`'s default constructor (which must
   * exist), and fills it with pseudo-random bytes.
   *
   * Some types that do not have default constructors may have specializations
   * defined.
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
   * Get a tuple of pseudo-random values within the specified bounds.
   *
   * This should be called like e.g.
   *    Rand.rand<int, short, unsigned>({min_for_int,      max_for_int},
   *                                    {min_for_short,    max_for_short},
   *                                    {min_for_unsigned, max_for_unsigned});
   */
  template <typename... Ts>
  std::tuple<Ts...> rand(std::tuple<Ts, Ts>... min_max_inclusive);

  /**
   * Get a pseudo-random value of type `T` using the supplied parameters.
   *
   * The idea here is that other classes may want to allow random generation of
   * objects subject to non-trivial constraints. Specializations of this
   * function can be defined elsewhere, using any appropriate parameter type
   * `T_params`
   */
  template <typename T, typename T_params>
  T rand(T_params params);

  void rand_bytes(void *dst, size_t size);
};

template <typename T>
T Rand::get_rand(Tag<T>) {
  if (sizeof(T) <= sizeof(int)) {
    return static_cast<T>(pseudo_rand());
  } else {
    T res;
    pseudo_rand_bytes(&res, sizeof(T));
    return res;
  }
}

template <typename... Ts>
std::tuple<Ts...> Rand::get_rand(Tag<std::tuple<Ts...> >) {
  return std::make_tuple<Ts...>(get_rand(Tag<Ts>())...);
}

template <typename T>
T Rand::get_rand(Tag<T>, T min, T max) {
  assert(min <= max);
  T span = max - min + 1;
  T x = static_cast<T>(((unsigned)pseudo_rand()) % span);
  return min + x;
}

template <int... indices, typename... Ts>
std::tuple<Ts...> Rand::get_rand(Tag<std::tuple<Ts...> >,
                                 index_list<indices...>, std::tuple<Ts...> min,
                                 std::tuple<Ts...> max) {
  return std::make_tuple<Ts...>(
      (this->rand<Ts>(std::get<indices>(min), std::get<indices>(max)))...);
}

template <typename... Ts>
std::tuple<Ts...> Rand::get_rand(Tag<std::tuple<Ts...> >, std::tuple<Ts...> min,
                                 std::tuple<Ts...> max) {
  return get_rand(Tag<std::tuple<Ts...> >(),
                  typename gen_indices<Ts...>::type(), min, max);
}

template <typename... Ts>
std::tuple<Ts...> Rand::get_rand_tuple(
    Tag<std::tuple<Ts...> >, std::tuple<Ts, Ts>... min_max_inclusive) {
  return std::make_tuple<Ts...>((this->rand<Ts>(
      std::get<0>(min_max_inclusive), std::get<1>(min_max_inclusive)))...);
}

template <typename T>
T Rand::rand() {
  return this->get_rand(Tag<T>());
}

template <typename T>
T Rand::rand(T min_inclusive, T max_inclusive) {
  return get_rand(Tag<T>(), min_inclusive, max_inclusive);
}

template <typename... Ts>
std::tuple<Ts...> Rand::rand(std::tuple<Ts, Ts>... min_max_inclusive) {
  return get_rand_tuple(Tag<std::tuple<Ts...> >(), min_max_inclusive...);
}

// Specializations of get_rand() defined in Rand.cpp need to be declared here or
// the compiler will choose the generic implementation.

template <>
int8_t Rand::get_rand<int8_t>(Tag<int8_t>);

// 50/50 true/false
template <>
bool Rand::get_rand<bool>(Tag<bool>);
// Uniform distribution over the range [-1.0f, 1.0f)
template <>
float Rand::get_rand<float>(Tag<float>);
// Uniform distribution over the range [-1.0, 1.0)
template <>
double Rand::get_rand<double>(Tag<double>);
// Uniform distribution over the range [max, max)
template <>
float Rand::get_rand<float>(Tag<float>, float, float);
// Uniform distribution over the range [min, max)
template <>
double Rand::get_rand<double>(Tag<double>, double, double);

template <typename T>
class RandIterBase {
 public:
  using RandType = T;

  class iterator : public std::iterator<std::input_iterator_tag, T> {
   private:
    RandIterBase *parent;
    unsigned dex;
    T value;

   public:
    explicit iterator(RandIterBase *parent, unsigned dex)
        : parent(parent), dex(dex), value(parent->NextRand()) {}

    iterator &operator++() {
      dex++;
      value = parent->NextRand();
      return *this;
    }

    iterator operator++(int) {
      iterator retval = *this;
      ++(*this);
      return retval;
    }
    bool operator==(iterator other) const { return this->dex == other.dex; }
    bool operator!=(iterator other) const { return this->dex != other.dex; }
    T &operator*() { return this->value; }
  };

  friend class iterator;

 protected:
  unsigned count;
  Rand rng;

  virtual T NextRand() = 0;

 public:
  RandIterBase(unsigned count) : count(count), rng(Rand()) {}

  constexpr RandIterBase(unsigned count, int seed) noexcept
      : count(count), rng(Rand(seed)) {}

  iterator begin() { return iterator(this, 0); }
  iterator end() { return iterator(this, count); }
};

template <typename T>
class RandIter : public RandIterBase<T> {
 protected:
  virtual T NextRand() override { return this->rng.template rand<T>(); }

 public:
  RandIter(unsigned count) : RandIterBase<T>(count) {}
  constexpr RandIter(unsigned count, int seed) noexcept
      : RandIterBase<T>(count, seed) {}
};

template <typename T_out, typename T_param>
class ParamedRandIter : public RandIterBase<T_out> {
 protected:
  T_param params;

  virtual T_out NextRand() override {
    return this->rng.template rand<T_out, T_param>(params);
  }

 public:
  ParamedRandIter(unsigned count, const T_param &params)
      : RandIterBase<T_out>(count), params(params) {}
  constexpr ParamedRandIter(unsigned count, const T_param &params,
                            int seed) noexcept
      : RandIterBase<T_out>(count, seed), params(params) {}
};

template <typename T>
class RandRangeIter : public RandIterBase<T> {
 protected:
  T min_value;
  T max_value;

  virtual T NextRand() override {
    return this->rng.template rand<T>(min_value, max_value);
  }

 public:
  RandRangeIter(unsigned count, T min, T max)
      : RandIterBase<T>(count), min_value(min), max_value(max) {}
  constexpr RandRangeIter(unsigned count, T min, T max, int seed) noexcept
      : RandIterBase<T>(count, seed), min_value(min), max_value(max) {}
};

template <typename... Types>
class RandRangeTupleIter : public RandIterBase<std::tuple<Types...> > {
  using RandType = std::tuple<Types...>;

 private:
  template <int... indices>
  RandType NextRand_(index_list<indices...>) {
    return this->rng.template rand<Types...>(std::get<indices>(bounds)...);
  }

 protected:
  std::tuple<std::tuple<Types, Types>...> bounds;

  virtual RandType NextRand() override {
    return NextRand_(typename gen_indices<Types...>::type());
  }

 public:
  RandRangeTupleIter(unsigned count,
                     std::tuple<Types, Types>... min_max_inclusive)
      : RandIterBase<RandType>(count), bounds(min_max_inclusive...) {}

  constexpr RandRangeTupleIter(unsigned count,
                               std::tuple<Types, Types>... min_max_inclusive,
                               int seed) noexcept
      : RandIterBase<RandType>(count, seed), bounds(min_max_inclusive...) {}
};

}  // namespace test
}  // namespace nn