#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "OutputTransformFn.hpp"
#include "Rand.hpp"

extern "C" {
#include "tst_common.h"
#include "unity.h"
}
using namespace nn;
using namespace nn::test;
static auto rng = test::Rand(42);

/*
  Test for zero multipliers
  Test for zero channels


*/

/*
  Given a number of coefficients this generates realistic accumulator min and
  max values.
*/
void pick_representitive_accu_bounds(int coef_count, int &accu_min,
                                     int &accu_max) {
  accu_min = 0;
  accu_max = 0;

  for (int i = 0; i < coef_count; i++) {
    int8_t b = rng.rand<int8_t>();

    if (b > 0) {
      accu_min += (int32_t)b * (int32_t)INT8_MIN;
      accu_max += (int32_t)b * (int32_t)INT8_MAX;
    } else {
      accu_min += (int32_t)b * (int32_t)INT8_MAX;
      accu_max += (int32_t)b * (int32_t)INT8_MIN;
    }
  }
}

void assert_word_aligned(const void *address) {
  assert(((uintptr_t)address & 0x3) == 0);
}

template <typename T>
T get_random_uniform_from_range(T lo, T hi) {
  return rng.rand<T>(lo, hi);
}

/*
coef_count - controls the number of coefs that make up the kernel.
N - controls the scale of the product, i.e. the product will be in the range
    [-1<<N, 1<<N - 1] but additionally the dynamic range is controlled by
    the product_range.
product_range - controls the dynamic range of the product.
bias_range - controls how much bigger or smaller the product should be
    compared to the product.
*/
void test_big_range(int coef_count, int N, int product_range, int bias_range,
                    std::set<int> &seen_initial_shr,
                    std::set<int> &seen_final_shr) {
  int64_t bias_low = -(1LL << (N + bias_range));
  int64_t bias_high = (1LL << (N + bias_range)) - 1;

  double p_low = -(1LL << std::max(N - product_range, 0));
  double p_high = (1LL << N) - 1;

  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  double error_sum = 0.0;
  double abs_error_sum = 0.0;
  int error_count = 0;

  for (int output_ch_count = 4; output_ch_count <= 64; output_ch_count += 4) {
    for (int itt = 0; itt < 8; itt++) {
      MulsAndBias mul_and_biases;
      for (int ch = 0; ch < output_ch_count; ++ch) {
        int accu_min, accu_max;
        // get the accu bounds for a given coef_count
        pick_representitive_accu_bounds(coef_count, accu_min, accu_max);

        // now pick an interesting bias an mul.

        // let's rescale all (accu * mul) products to [-2**N, 2**N-1]

        // pick a number between 2**(N-8)-1 and 2**N-1

        float product_target = get_random_uniform_from_range(p_low, p_high);

        float multiplier =
            std::abs(product_target / (float)std::max(-accu_min, accu_max));

        // the bias must be between [-2**(N+1) - 1, 2**(N+1)]
        double bias =
            (double)get_random_uniform_from_range(bias_low, bias_high);

        OutputTransformFn::ActivationParams a(bias, multiplier, accu_min,
                                              accu_max);
        mul_and_biases.push_back(a);
      }
      auto quantizer = OutputTransformFnInt8_Group::Quantizer();
      OutputTransformFnInt8_Group::QuantisationParams qp =
          quantizer.quantise_activation(mul_and_biases, false);

      seen_final_shr.insert(qp.final_shr);
      seen_initial_shr.insert(qp.initial_shr);

      // pad q.multipliers_and_biases to a multiple of VPU_INT16_EPV
      // this is to work around array over reads - padding wont effect the
      // result.
      int16_t pad_val = rng.rand<int16_t>();

      auto serialised_multipliers_and_biases =
          OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);

      OutputTransformFn::pad_final_access(serialised_multipliers_and_biases,
                                          VPU_INT16_EPV, pad_val);

      OT_int8 ot((int32_t)output_ch_count, qp.initial_shr, qp.final_shr);
      otfn_int8_params_t p = ot.getParams();

      int8_t Y[output_ch_count];
      memset(Y, 0, sizeof Y);

      int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) /
                      vpu_ring_buffer_length;

      int8_t *y = (int8_t *)Y;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        int chs_in_group =
            std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                     vpu_ring_buffer_length);

        VPURingBuffer A;

        int8_t *next_y;

        for (int t = 0; t < 1 << 6; t++) {
          memset(&A, 0, sizeof A);

          int32_t accu_values[chs_in_group];

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int actual_output_channel =
                output_chan + ocg * vpu_ring_buffer_length;

            int32_t accu_min =
                mul_and_biases[actual_output_channel].original_accu_min_val;
            int32_t accu_max =
                mul_and_biases[actual_output_channel].original_accu_max_val;

            int32_t v = rng.rand<int32_t>(accu_min, accu_max);

            accu_values[output_chan] = v;
            A.vR[output_chan] = ((int16_t *)&v)[0];
            A.vD[output_chan] = ((int16_t *)&v)[1];
          }

          next_y = otfn_int8(&p, y, &A, ocg, serialised_multipliers_and_biases.data());

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int actual_output_channel =
                output_chan + ocg * vpu_ring_buffer_length;

            double expected =
                (double)accu_values[output_chan] *
                    mul_and_biases[actual_output_channel].original_multiplier +
                mul_and_biases[actual_output_channel].original_bias;

            expected = std::min(std::max(expected, (double)INT8_MIN),
                                           (double)INT8_MAX);

            int actual = (int)Y[actual_output_channel];

            TEST_ASSERT_INT32_WITHIN(1, (int)std::round(expected), actual);

            error_count += 1;
            error_sum += (expected - actual);
            abs_error_sum += std::abs(expected - actual);
          }
        }
        y = next_y;
      }
    }
  }
  float bias = error_sum / error_count;

  TEST_ASSERT_TRUE_MESSAGE(std::abs(bias) < 0.025, "Bias out of range");

  TEST_ASSERT_TRUE_MESSAGE((abs_error_sum/error_count) < (0.25 + 0.001), //the extra bit is to account for random sampling
                           "abs average error too high");
}

void test_big_range_channelwise(int coef_count, int N, int product_range,
                                int bias_range, std::set<int> &seen_initial_shr,
                                std::set<int> &seen_final_shr) {
  int64_t bias_low = -(1LL << (N + bias_range));
  int64_t bias_high = (1LL << (N + bias_range)) - 1;

  double p_low = -(1LL << std::max(N - product_range, 0));
  double p_high = (1LL << N) - 1;

  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  double error_sum = 0.0;
  double abs_error_sum = 0.0;
  int error_count = 0;

  for (int output_ch_count = 4; output_ch_count <= 64; output_ch_count += 4) {
    for (int itt = 0; itt < 8; itt++) {
      MulsAndBias mul_and_biases;
      for (int ch = 0; ch < output_ch_count; ++ch) {
        int accu_min, accu_max;
        // get the accu bounds for a given coef_count
        pick_representitive_accu_bounds(coef_count, accu_min, accu_max);

        // now pick an interesting bias and mul.

        // let's rescale all (accu * mul) products to [-2**N, 2**N-1]

        // pick a number between 2**(N-8)-1 and 2**N-1

        float product_target = get_random_uniform_from_range(p_low, p_high);

        float multiplier =
            product_target / (float)std::max(-accu_min, accu_max);

        // the bias must be between [-2**(N+1) - 1, 2**(N+1)]
        double bias =
            (double)get_random_uniform_from_range(bias_low, bias_high);

        OutputTransformFn::ActivationParams a(bias, multiplier, accu_min,
                                              accu_max);
        mul_and_biases.push_back(a);
      }

      auto quantizer = OutputTransformFnInt8_Channelwise::Quantizer();
      OutputTransformFnInt8_Channelwise::QuantisationParams qp =
          quantizer.quantise_activation(mul_and_biases, false);

      seen_final_shr.insert(qp.final_shr);
      seen_initial_shr.insert(qp.initial_shifts[0]);

      // pad q.multipliers_and_biases to a multiple of VPU_INT16_EPV
      // this is to work around array over reads - padding wont effect the
      // result.
      int16_t pad_val = rng.rand<int16_t>();

      auto serialised_multipliers_and_biases =
          OutputTransformFn::serialise_memory(qp.initial_shifts, qp.multipliers,
                                              qp.biases);

      OutputTransformFn::pad_final_access(serialised_multipliers_and_biases,
                                          VPU_INT16_EPV, pad_val);

      OT_int8_channelwise ot((int32_t)output_ch_count, qp.final_shr);
      otfn_int8_channelwise_params_t p = ot.getParams();

      int8_t Y[output_ch_count];
      memset(Y, 0, sizeof Y);

      int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) /
                      vpu_ring_buffer_length;

      int8_t *y = (int8_t *)Y;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        int chs_in_group =
            std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                     vpu_ring_buffer_length);

        VPURingBuffer A;

        int8_t *next_y;

        for (int t = 0; t < 1 << 6; t++) {
          memset(&A, 0, sizeof A);

          int32_t accu_values[chs_in_group];

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int actual_output_channel =
                output_chan + ocg * vpu_ring_buffer_length;

            int32_t accu_min =
                mul_and_biases[actual_output_channel].original_accu_min_val;
            int32_t accu_max =
                mul_and_biases[actual_output_channel].original_accu_max_val;

            int32_t v = rng.rand<int32_t>(accu_min, accu_max);

            accu_values[output_chan] = v;
            A.vR[output_chan] = ((int16_t *)&v)[0];
            A.vD[output_chan] = ((int16_t *)&v)[1];
          }

          next_y = otfn_int8_channelwise(&p, y, &A, ocg, serialised_multipliers_and_biases.data());
          if (qp.final_shr <= 0) {
            for (int output_chan = 0; output_chan < chs_in_group;
                 ++output_chan) {
              int actual_output_channel =
                  output_chan + ocg * vpu_ring_buffer_length;

              double expected =
                  (double)accu_values[output_chan] *
                      mul_and_biases[actual_output_channel]
                          .original_multiplier +
                  mul_and_biases[actual_output_channel].original_bias;

              expected = std::round(std::min(
                  std::max(expected, (double)INT8_MIN), (double)INT8_MAX));

              int actual = (int)Y[actual_output_channel];
              TEST_ASSERT_INT32_WITHIN(1, (int)expected, actual);

              error_count += 1;
              error_sum += (expected - actual);
              abs_error_sum += std::abs(expected - actual);
            }
          }
        }
        y = next_y;
      }
    }
  }
  float bias = error_sum / error_count;

  // TEST_ASSERT_TRUE_MESSAGE(std::abs(bias) < 2e-2, "Bias out of range");
  // printf("bias %d, error_count %d, error_sum %d, average error %d\n", bias,
  // error_count, abs_error_sum, (error_count/abs_error_sum));
  // TEST_ASSERT_TRUE_MESSAGE(error_count / abs_error_sum > 100,
  //                          "abs average error too high");
}

/*
  All channels are the same. Multiplier is always 1.0.
*/
void test_small_range(const int accu_min, const int accu_max,
                      const int const_bias) {
  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  for (int output_ch_count = 4; output_ch_count <= 64; output_ch_count += 4) {
    MulsAndBias mul_and_biases;

    for (int ch = 0; ch < output_ch_count; ++ch) {
      OutputTransformFn::ActivationParams a(const_bias, 1.0, accu_min,
                                            accu_max);
      mul_and_biases.push_back(a);
    }

    auto quantizer = OutputTransformFnInt8_Group::Quantizer();
    OutputTransformFnInt8_Group::QuantisationParams qp =
        quantizer.quantise_activation(mul_and_biases, false);

    auto serialised_multipliers_and_biases =
        OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);

    // pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
    // this is to work around array over reads
    int16_t pad_val = rng.rand<int16_t>();  // this is arbitrary

    OutputTransformFn::pad_final_access(serialised_multipliers_and_biases,
                                        VPU_INT16_EPV, pad_val);

    OT_int8 ot((int32_t)output_ch_count, qp.initial_shr, qp.final_shr);
    otfn_int8_params_t p = ot.getParams();

    int8_t Y[output_ch_count];
    memset(Y, 0, sizeof Y);

    int ocg_count =
        (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

    int8_t *y = (int8_t *)Y;

    for (int ocg = 0; ocg < ocg_count; ++ocg) {
      int chs_in_group =
          std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                   vpu_ring_buffer_length);

      VPURingBuffer A;

      int8_t *next_y;

      for (int t = accu_min; t <= accu_max; t++) {
        memset(&A, 0, sizeof A);

        int32_t accu_values[chs_in_group];
        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int32_t v = t;
          accu_values[output_chan] = v;
          A.vR[output_chan] = ((int16_t *)&v)[0];
          A.vD[output_chan] = ((int16_t *)&v)[1];
        }

        next_y = otfn_int8(&p, y, &A, ocg, serialised_multipliers_and_biases.data());

        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int actual_output_channel =
              output_chan + ocg * vpu_ring_buffer_length;

          double expected =
              (float)t *
                  mul_and_biases[actual_output_channel].original_multiplier +
              mul_and_biases[actual_output_channel].original_bias;
          expected = std::round(
              std::min(std::max(expected, (double)INT8_MIN), (double)INT8_MAX));

          TEST_ASSERT_INT32_WITHIN(1, (int)expected,
                                   (int)Y[actual_output_channel]);
        }
      }
      y = next_y;
    }
  }
}

/*
  All channels are the same. Multiplier is always 1.0.
*/
void test_small_range_channelwise(const int accu_min, const int accu_max,
                                  const int const_bias) {
  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  for (int output_ch_count = 4; output_ch_count <= 64; output_ch_count += 4) {
    MulsAndBias mul_and_biases;

    for (int ch = 0; ch < output_ch_count; ++ch) {
      OutputTransformFn::ActivationParams a(const_bias, 1.0, accu_min,
                                            accu_max);
      mul_and_biases.push_back(a);
    }

    auto quantizer = OutputTransformFnInt8_Channelwise::Quantizer();
    OutputTransformFnInt8_Channelwise::QuantisationParams qp =
        quantizer.quantise_activation(mul_and_biases, false);

    auto serialised_multipliers_and_biases =
        OutputTransformFn::serialise_memory(qp.initial_shifts, qp.multipliers,
                                            qp.biases);

    // pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
    // this is to work around array over reads
    int16_t pad_val = rng.rand<int16_t>();  // this is arbitrary

    OutputTransformFn::pad_final_access(serialised_multipliers_and_biases,
                                        VPU_INT16_EPV, pad_val);

    OT_int8_channelwise ot((int32_t)output_ch_count, qp.final_shr);
    otfn_int8_channelwise_params_t p = ot.getParams();

    int8_t Y[output_ch_count];
    memset(Y, 0, sizeof Y);

    int ocg_count =
        (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

    int8_t *y = (int8_t *)Y;

    for (int ocg = 0; ocg < ocg_count; ++ocg) {
      int chs_in_group =
          std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                   vpu_ring_buffer_length);

      VPURingBuffer A;

      int8_t *next_y;

      for (int t = accu_min; t <= accu_max; t++) {
        memset(&A, 0, sizeof A);

        int32_t accu_values[chs_in_group];
        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int32_t v = t;
          accu_values[output_chan] = v;
          A.vR[output_chan] = ((int16_t *)&v)[0];
          A.vD[output_chan] = ((int16_t *)&v)[1];
        }

        next_y = otfn_int8_channelwise(&p, y, &A, ocg, serialised_multipliers_and_biases.data());

        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int actual_output_channel =
              output_chan + ocg * vpu_ring_buffer_length;

          double expected =
              (float)t *
                  mul_and_biases[actual_output_channel].original_multiplier +
              mul_and_biases[actual_output_channel].original_bias;
          expected = std::round(
              std::min(std::max(expected, (double)INT8_MIN), (double)INT8_MAX));

          TEST_ASSERT_INT32_WITHIN(1, (int)expected,
                                   (int)Y[actual_output_channel]);
        }
      }
      y = next_y;
    }
  }
}

void Test_OT_int8_range() { test_small_range(INT8_MIN, INT8_MAX, 0); }
void Test_OT_int8_channelwise_range() {
  test_small_range_channelwise(INT8_MIN, INT8_MAX, 0);
}

void Test_OT_int8_range_small_bias() {
  test_small_range(INT8_MIN, INT8_MAX, 1);
}

void Test_OT_int8_channelwise_range_small_bias() {
  test_small_range_channelwise(INT8_MIN, INT8_MAX, 1);
}

void Test_OT_int8_range_small_bias2() {
  test_small_range(INT8_MIN, INT8_MAX, -1);
}

void Test_OT_int8_channelwise_range_small_bias2() {
  test_small_range_channelwise(INT8_MIN, INT8_MAX, -1);
}

void Test_OT_int8_small_range() {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, 0);
}

void Test_OT_int8_channelwise_small_range() {
  test_small_range_channelwise(INT8_MIN - 10, INT8_MAX + 10, 0);
}

void Test_OT_int8_small_range_bias() {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, 1);
}
void Test_OT_int8_channelwise_small_range_bias() {
  test_small_range_channelwise(INT8_MIN - 10, INT8_MAX + 10, 1);
}
void Test_OT_int8_small_range_bias2() {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, -1);
}
void Test_OT_int8_channelwise_small_range_bias2() {
  test_small_range_channelwise(INT8_MIN - 10, INT8_MAX + 10, -1);
}

void Test_OT_int8_small_range_wide_bias_range() {
  for (int bias = INT8_MIN * 2 - 10; bias < 48; bias++) {
    test_small_range(INT8_MIN, INT8_MAX, bias);
  }
}

void Test_OT_int8_channelwise_small_range_wide_bias_range() {
  for (int bias = INT8_MIN * 2 - 10; bias < 48; bias++) {
    test_small_range_channelwise(INT8_MIN, INT8_MAX, bias);
  }
}

void Test_OT_int8_small_range_massive_bias_range() {
  for (int itt = 0; itt < 32; ++itt) {
    int32_t bias = rng.rand<int32_t>(1 << 15, 1 << 30);
    test_small_range(INT8_MIN, INT8_MAX, bias);
  }
}

void Test_OT_int8_channelwise_small_range_massive_bias_range() {
  for (int itt = 0; itt < 32; ++itt) {
    int32_t bias = rng.rand<int32_t>(1 << 15, 1 << 30);
    test_small_range_channelwise(INT8_MIN, INT8_MAX, bias);
  }
}

void Test_OT_int8_big_range() {
  std::set<int> seen_initial_shift;
  std::set<int> seen_final_shr;
  for (int product_range = -3; product_range < 3; product_range++) {
    for (int n = 3; n < 9; n++) {
      for (int bias_range = -n; bias_range < 3; bias_range++) {
        for (int coef_count_log2 = 2; coef_count_log2 < 12;
             coef_count_log2 += 1) {
          test_big_range(1 << coef_count_log2, n, product_range, bias_range,
                         seen_initial_shift, seen_final_shr);
        }
      }
    }
  }
#define INITIAL_SHR_RANGE_MAX 12
#define INITIAL_SHR_RANGE_MIN 0

#define FINAL_SHR_RANGE_MAX 6
#define FINAL_SHR_RANGE_MIN -8

  for (int i = INITIAL_SHR_RANGE_MIN; i <= INITIAL_SHR_RANGE_MAX; i++) {
    const bool is_in = seen_initial_shift.find(i) != seen_initial_shift.end();
    // TEST_ASSERT_TRUE(is_in);
  }
  for (int i = FINAL_SHR_RANGE_MIN; i <= FINAL_SHR_RANGE_MAX; i++) {
    const bool is_in = seen_final_shr.find(i) != seen_final_shr.end();
    // TEST_ASSERT_TRUE(is_in);
  }
}

void Test_OT_int8_channelwise_big_range() {
  std::set<int> seen_initial_shift;
  std::set<int> seen_final_shr;
  for (int product_range = -3; product_range < 3; product_range++) {
    for (int n = 3; n < 9; n++) {
      for (int bias_range = -n; bias_range < 3; bias_range++) {
        for (int coef_count_log2 = 2; coef_count_log2 < 12;
             coef_count_log2 += 1) {
          test_big_range_channelwise(1 << coef_count_log2, n, product_range,
                                     bias_range, seen_initial_shift,
                                     seen_final_shr);
        }
      }
    }
  }
#define INITIAL_SHR_RANGE_MAX 12
#define INITIAL_SHR_RANGE_MIN 0

#define FINAL_SHR_RANGE_MAX 6
#define FINAL_SHR_RANGE_MIN -8

  // for(auto ini_shift : seen_initial_shift) printf("seen initial shift: %d\n",
  // ini_shift); for(auto ini_shift : seen_final_shr) printf("seen final shift:
  // %d\n", ini_shift);
  for (int i = INITIAL_SHR_RANGE_MIN; i <= INITIAL_SHR_RANGE_MAX; i++) {
    const bool is_in = seen_initial_shift.find(i) != seen_initial_shift.end();

    // TEST_ASSERT_TRUE(is_in);
  }
  for (int i = FINAL_SHR_RANGE_MIN; i <= FINAL_SHR_RANGE_MAX; i++) {
    const bool is_in = seen_final_shr.find(i) != seen_final_shr.end();
    // TEST_ASSERT_TRUE(is_in);
  }
}

extern "C" void test_output_transforms();
void test_output_transforms() {
  UNITY_SET_FILE();
  RUN_TEST(Test_OT_int8_range);
  RUN_TEST(Test_OT_int8_channelwise_range);
  RUN_TEST(Test_OT_int8_range_small_bias);
  RUN_TEST(Test_OT_int8_channelwise_range_small_bias);
  RUN_TEST(Test_OT_int8_range_small_bias2);
  RUN_TEST(Test_OT_int8_channelwise_range_small_bias2);
  RUN_TEST(Test_OT_int8_small_range);
  RUN_TEST(Test_OT_int8_channelwise_small_range);
  RUN_TEST(Test_OT_int8_small_range_bias);
  RUN_TEST(Test_OT_int8_channelwise_small_range_bias);
  RUN_TEST(Test_OT_int8_small_range_bias2);
  RUN_TEST(Test_OT_int8_channelwise_small_range_bias2);
  RUN_TEST(Test_OT_int8_small_range_massive_bias_range);
  RUN_TEST(Test_OT_int8_channelwise_small_range_massive_bias_range);
  RUN_TEST(Test_OT_int8_small_range_wide_bias_range);
  RUN_TEST(Test_OT_int8_channelwise_small_range_wide_bias_range);
  RUN_TEST(Test_OT_int8_big_range);
  RUN_TEST(Test_OT_int8_channelwise_big_range);
}
