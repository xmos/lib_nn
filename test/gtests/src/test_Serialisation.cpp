#include <cstring>
#include <vector>

#include "AbstractKernel.hpp"
#include "AggregateFn.hpp"
#include "MemCpyFn.hpp"
#include "OutputTransformFn.hpp"
#include "Rand.hpp"
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
    T *q = Serialisable::deserialise<T>(s.c_str());

    for (size_t i = 0; i < sizeof(T); i++) {
      EXPECT_EQ(((int8_t *)p)[i], ((int8_t *)q)[i]);
    }
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

class Test_MatMulDirectFnParams : public ::testing::Test {};

TEST_F(Test_MatMulDirectFnParams, BasicTest) {
  test_serialisation<MatMulDirectFn::Params>();
}

class Test_OT_int8Params : public ::testing::Test {};

TEST_F(Test_OT_int8Params, BasicTest) {
  int8_t d[sizeof(OT_int8::Params)];

  for (int t = 0; t < test_count; ++t) {
    int output_slice_channel_count = t;
    std::vector<int16_t> biases(output_slice_channel_count);
    std::vector<int16_t> multipliers(output_slice_channel_count);
    OutputTransformValues otv;

    for (int idx = 0; idx < output_slice_channel_count; ++idx) {
      biases[idx] = rng.rand<int16_t>();
      multipliers[idx] = rng.rand<int16_t>();
    }
    for (int idx = 0; idx < sizeof(OutputTransformValues); ++idx) {
      ((int8_t *)&otv)[idx] = rng.rand<int8_t>();
    }

    OT_int8::Params p(output_slice_channel_count, &otv, biases, multipliers);
    std::string s = p.template serialise<OT_int8::Params>();
    OT_int8::Params *q = Serialisable::deserialise<OT_int8::Params>(s.c_str());

    EXPECT_EQ(p.output_slice_channel_count, q->output_slice_channel_count);

    for (size_t i = 0; i < sizeof(OutputTransformValues); i++) {
      EXPECT_EQ(((int8_t *)p.otv)[i], ((int8_t *)q->otv)[i]);
    }
    for (size_t i = 0; i < output_slice_channel_count; i++) {
      EXPECT_EQ(p.biases[i], q->biases[i]);
      EXPECT_EQ(p.multipliers[i], q->multipliers[i]);
    }
  }
}