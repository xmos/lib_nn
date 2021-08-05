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
  int8_t d[sizeof(MatMulInt8::Params)];

  for (int t = 0; t < test_count; ++t) {
    for (int idx = 0; idx < sizeof(d); ++idx) d[idx] = rng.rand<int8_t>();

    MatMulInt8::Params *p = (MatMulInt8::Params *)d;

    p->bytes_per_kernel_channel = (int)rng.rand<uint8_t>();
    p->output_slice_channel_count = (int)rng.rand<uint8_t>();

    int weight_count = MatMulInt8::get_weights_bytes(
        p->bytes_per_kernel_channel, p->output_slice_channel_count);

    std::vector<int8_t> weights(weight_count);

    for (int idx = 0; idx < weight_count; ++idx)
      weights[idx] = rng.rand<int8_t>();

    p->weights = (int8_t *)weights.data();

    std::string s = p->template serialise<MatMulInt8::Params>();

    int allocation_byte_count =
        MatMulInt8::Params::get_allocation_byte_count(s.c_str()) +
        sizeof(MatMulInt8::Params);
    char *allocated_memory = (char *)std::malloc(allocation_byte_count);

    MatMulInt8::Params *q = MatMulInt8::Params::deserialise<MatMulInt8::Params>(
        allocated_memory, s.c_str());

    EXPECT_NE(p, q);
    EXPECT_EQ(p->output_slice_channel_count, q->output_slice_channel_count);
    EXPECT_EQ(p->bytes_per_kernel_channel, q->bytes_per_kernel_channel);

    for (size_t i = 0; i < weight_count; i++) {
      EXPECT_EQ(p->weights[i], q->weights[i]);
    }
    free(allocated_memory);
  }
}

class Test_MatMulDirectFnParams : public ::testing::Test {};

TEST_F(Test_MatMulDirectFnParams, BasicTest) {
  int8_t d[sizeof(MatMulDirectFn::Params)];

  for (int t = 0; t < test_count; ++t) {
    int weight_bytes = t;

    for (int idx = 0; idx < sizeof(d); ++idx) d[idx] = rng.rand<int8_t>();

    std::vector<int8_t> weights(weight_bytes);

    for (int idx = 0; idx < weight_bytes; ++idx)
      weights[idx] = rng.rand<int8_t>();

    MatMulDirectFn::Params *p = (MatMulDirectFn::Params *)d;
    p->weights = (int8_t *)weights.data();
    p->weights_bytes = weight_bytes;

    std::string s = p->template serialise<MatMulDirectFn::Params>();

    int allocation_byte_count =
        MatMulDirectFn::Params::get_allocation_byte_count(s.c_str()) +
        sizeof(MatMulDirectFn::Params);
    char *allocated_memory = (char *)std::malloc(allocation_byte_count);

    MatMulDirectFn::Params *q =
        MatMulDirectFn::Params::deserialise<MatMulDirectFn::Params>(
            allocated_memory, s.c_str());

    EXPECT_NE(p, q);
    EXPECT_EQ(p->bytes_per_kernel_channel, q->bytes_per_kernel_channel);
    EXPECT_EQ(p->k_height_loop_counter, q->k_height_loop_counter);
    EXPECT_EQ(p->k_width_loop_counter, q->k_width_loop_counter);
    EXPECT_EQ(p->inner_x_h_step, q->inner_x_h_step);
    EXPECT_EQ(p->inner_x_v_step, q->inner_x_v_step);
    EXPECT_EQ(p->weights_bytes, q->weights_bytes);
    EXPECT_NE(p->weights, q->weights);

    for (size_t i = 0; i < weight_bytes; i++) {
      EXPECT_EQ(p->weights[i], q->weights[i]);
    }
    free(allocated_memory);
  }
}

class Test_OT_int8Params : public ::testing::Test {};

TEST_F(Test_OT_int8Params, BasicTest) {
  for (int t = 1; t < test_count; ++t) {
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

    int allocation_byte_count =
        OT_int8::Params::get_allocation_byte_count(s.c_str()) +
        sizeof(OT_int8::Params);

    char *allocated_memory = (char *)std::malloc(allocation_byte_count);

    OT_int8::Params *q = OT_int8::Params::deserialise<OT_int8::Params>(
        allocated_memory, s.c_str());

    EXPECT_EQ(p.output_slice_channel_count, q->output_slice_channel_count);
    EXPECT_EQ(p.mul_and_bias_size, q->mul_and_bias_size);
    EXPECT_NE(p.otv, q->otv);
    EXPECT_NE(p.biases, q->biases);
    EXPECT_NE(p.multipliers, q->multipliers);

    for (size_t i = 0; i < sizeof(OutputTransformValues); i++) {
      EXPECT_EQ(((int8_t *)p.otv)[i], ((int8_t *)q->otv)[i]);
    }
    for (size_t i = 0; i < output_slice_channel_count; i++) {
      EXPECT_EQ(p.biases[i], q->biases[i]);
      EXPECT_EQ(p.multipliers[i], q->multipliers[i]);
    }
    free(allocated_memory);
  }
}