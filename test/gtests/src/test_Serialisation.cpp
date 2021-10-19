#include <cstring>
#include <vector>

#include "AbstractKernel.hpp"
#include "AggregateFn.hpp"
#include "MemCpyFn.hpp"
#include "OutputTransformFn.hpp"
#include "Rand.hpp"
#include "Serialisable.hpp"
#include "gtest/gtest.h"

using namespace nn;
using namespace nn::test;
static auto rng = Rand(69);

static const int test_count = 1 << 10;

template <class T>
void test_serialisation() {
  int8_t d[sizeof(T)];

  for (int t = 0; t < test_count; ++t) {
    for (int idx = 0; idx < sizeof(T); ++idx) d[idx] = rng.rand<int8_t>();

    T *p = (T *)d;
    std::string s = p->template serialise<T>();

    char *allocated_memory = (char *)std::malloc(sizeof(T));

    T *q = Serialisable::deserialise<T>(allocated_memory, s.c_str());

    for (size_t i = 0; i < sizeof(T); i++) {
      EXPECT_EQ(((int8_t *)p)[i], ((int8_t *)q)[i]);
    }
    free(allocated_memory);
  }
}

class Test_AbstractKernelParams : public ::testing::Test {};

TEST_F(Test_AbstractKernelParams, BasicTest) {
  test_serialisation<AbstractKernel::Params>();
}

class Test_ImToColValidParams : public ::testing::Test {};

TEST_F(Test_ImToColValidParams, BasicTest) {
  test_serialisation<ImToColValid::Params>();
}

class Test_ImToColPaddedParams : public ::testing::Test {};

TEST_F(Test_ImToColPaddedParams, BasicTest) {
  test_serialisation<ImToColPadded::Params>();
}

class Test_DerefInputFnParams : public ::testing::Test {};

TEST_F(Test_DerefInputFnParams, BasicTest) {
  test_serialisation<DerefInputFn::Params>();
}

class Test_MatMulInt8Params : public ::testing::Test {};

TEST_F(Test_MatMulInt8Params, BasicTest) {
  test_serialisation<MatMulInt8::Params>();
}

class Test_OT_int8Params : public ::testing::Test {};

TEST_F(Test_OT_int8Params, BasicTest) { test_serialisation<OT_int8::Params>(); }
