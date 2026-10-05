// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <algorithm>
#include <cmath>
#include <random>
#include <set>

#include "OutputTransformFn.hpp"
#include "Rand.hpp"

extern "C" {
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"
}
using namespace nn;
using namespace nn::test;
static auto rng = test::Rand(42);

extern "C" {

TEST_GROUP(group_output_transforms);
TEST_SETUP(group_output_transforms) {}
TEST_TEAR_DOWN(group_output_transforms) {}
TEST_GROUP_RUNNER(group_output_transforms) {
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_range_small_bias);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_range_small_bias);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_range_small_bias2);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_range_small_bias2);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_small_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_small_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_small_range_bias);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_small_range_bias);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_small_range_bias2);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_small_range_bias2);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_small_range_massive_bias_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_small_range_massive_bias_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_small_range_wide_bias_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_small_range_wide_bias_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_big_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_big_range);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_bias_low_precision);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_channelwise_bias_low_precision);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_bias_left_shifted_accu);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_zero_multiplier);
  RUN_TEST_CASE(group_output_transforms, Test_OT_int8_bias_narrow_range);
}

}  // extern "C"

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
          quantizer.quantise_activation(mul_and_biases, nn_vlmul_shr_t::VLMUL_SHR_XS3A, false);

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

      std::vector<int8_t> vector_y(output_ch_count);
      std::fill(vector_y.begin(), vector_y.end(), 0);
      int8_t* y = vector_y.data();
      int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        int chs_in_group =
            std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                     vpu_ring_buffer_length);

        VPURingBuffer A;

        int8_t *next_y;

        for (int t = 0; t < 1 << 6; t++) {
          memset(&A, 0, sizeof A);

          std::vector<int32_t> accu_values_vector(chs_in_group);
          int32_t* accu_values = accu_values_vector.data();

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int actual_output_channel = output_chan + ocg * vpu_ring_buffer_length;
            int32_t accu_min = mul_and_biases[actual_output_channel].original_accu_min_val;
            int32_t accu_max = mul_and_biases[actual_output_channel].original_accu_max_val;
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

            int actual = (int)vector_y[actual_output_channel];

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
          quantizer.quantise_activation(mul_and_biases, nn_vlmul_shr_t::VLMUL_SHR_XS3A, false);

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

      std::vector<int8_t> vector_y(output_ch_count);
      std::fill(vector_y.begin(), vector_y.end(), 0);
      int8_t* y = vector_y.data();
      int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        int chs_in_group =
            std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                     vpu_ring_buffer_length);

        VPURingBuffer A;

        int8_t *next_y;

        for (int t = 0; t < 1 << 6; t++) {
          memset(&A, 0, sizeof A);

          std::vector<int32_t> accu_values_vector(chs_in_group);
          int32_t* accu_values = accu_values_vector.data();

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

              int actual = (int)vector_y[actual_output_channel];
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
  (void)bias;
  // TODO: check bias ranges and accepted values, uncomment below
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
        quantizer.quantise_activation(mul_and_biases, nn_vlmul_shr_t::VLMUL_SHR_XS3A, false);

    auto serialised_multipliers_and_biases =
        OutputTransformFn::serialise_memory(qp.multipliers, qp.biases);

    // pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
    // this is to work around array over reads
    int16_t pad_val = rng.rand<int16_t>();  // this is arbitrary

    OutputTransformFn::pad_final_access(serialised_multipliers_and_biases,
                                        VPU_INT16_EPV, pad_val);

    OT_int8 ot((int32_t)output_ch_count, qp.initial_shr, qp.final_shr);
    otfn_int8_params_t p = ot.getParams();

    std::vector<int8_t> vector_y(output_ch_count);
    std::fill(vector_y.begin(), vector_y.end(), 0);
    int8_t* y = vector_y.data();
    int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

    for (int ocg = 0; ocg < ocg_count; ++ocg) {
      int chs_in_group =
          std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                   vpu_ring_buffer_length);

      VPURingBuffer A;

      int8_t *next_y;

      for (int t = accu_min; t <= accu_max; t++) {
        memset(&A, 0, sizeof A);

        std::vector<int32_t> accu_values_vector(chs_in_group);
        int32_t* accu_values = accu_values_vector.data();
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
                                   (int)vector_y[actual_output_channel]);
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
        quantizer.quantise_activation(mul_and_biases, nn_vlmul_shr_t::VLMUL_SHR_XS3A, false);

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

    std::vector<int8_t> vector_y(output_ch_count);
    std::fill(vector_y.begin(), vector_y.end(), 0);
    int8_t* y = vector_y.data();
    int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;
    
    for (int ocg = 0; ocg < ocg_count; ++ocg) {
      int chs_in_group =
          std::min(output_ch_count - vpu_ring_buffer_length * ocg,
                   vpu_ring_buffer_length);

      VPURingBuffer A;

      int8_t *next_y;

      for (int t = accu_min; t <= accu_max; t++) {
        memset(&A, 0, sizeof A);

        std::vector<int32_t> accu_values_vector(chs_in_group);
        int32_t* accu_values = accu_values_vector.data();
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
                                   (int)vector_y[actual_output_channel]);
        }
      }
      y = next_y;
    }
  }
}

extern "C" {

TEST(group_output_transforms, Test_OT_int8_range) { test_small_range(INT8_MIN, INT8_MAX, 0); }
TEST(group_output_transforms, Test_OT_int8_channelwise_range) {
  test_small_range_channelwise(INT8_MIN, INT8_MAX, 0);
}

TEST(group_output_transforms, Test_OT_int8_range_small_bias) {
  test_small_range(INT8_MIN, INT8_MAX, 1);
}

TEST(group_output_transforms, Test_OT_int8_channelwise_range_small_bias) {
  test_small_range_channelwise(INT8_MIN, INT8_MAX, 1);
}

TEST(group_output_transforms, Test_OT_int8_range_small_bias2) {
  test_small_range(INT8_MIN, INT8_MAX, -1);
}

TEST(group_output_transforms, Test_OT_int8_channelwise_range_small_bias2) {
  test_small_range_channelwise(INT8_MIN, INT8_MAX, -1);
}

TEST(group_output_transforms, Test_OT_int8_small_range) {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, 0);
}

TEST(group_output_transforms, Test_OT_int8_channelwise_small_range) {
  test_small_range_channelwise(INT8_MIN - 10, INT8_MAX + 10, 0);
}

TEST(group_output_transforms, Test_OT_int8_small_range_bias) {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, 1);
}
TEST(group_output_transforms, Test_OT_int8_channelwise_small_range_bias) {
  test_small_range_channelwise(INT8_MIN - 10, INT8_MAX + 10, 1);
}
TEST(group_output_transforms, Test_OT_int8_small_range_bias2) {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, -1);
}
TEST(group_output_transforms, Test_OT_int8_channelwise_small_range_bias2) {
  test_small_range_channelwise(INT8_MIN - 10, INT8_MAX + 10, -1);
}

TEST(group_output_transforms, Test_OT_int8_small_range_wide_bias_range) {
  for (int bias = INT8_MIN * 2 - 10; bias < 48; bias++) {
    test_small_range(INT8_MIN, INT8_MAX, bias);
  }
}

TEST(group_output_transforms,
     Test_OT_int8_channelwise_small_range_wide_bias_range) {
  for (int bias = INT8_MIN * 2 - 10; bias < 48; bias++) {
    test_small_range_channelwise(INT8_MIN, INT8_MAX, bias);
  }
}

TEST(group_output_transforms, Test_OT_int8_small_range_massive_bias_range) {
  for (int itt = 0; itt < 32; ++itt) {
    int32_t bias = rng.rand<int32_t>(1 << 15, 1 << 30);
    test_small_range(INT8_MIN, INT8_MAX, bias);
  }
}

TEST(group_output_transforms,
     Test_OT_int8_channelwise_small_range_massive_bias_range) {
  for (int itt = 0; itt < 32; ++itt) {
    int32_t bias = rng.rand<int32_t>(1 << 15, 1 << 30);
    test_small_range_channelwise(INT8_MIN, INT8_MAX, bias);
  }
}

TEST(group_output_transforms, Test_OT_int8_big_range) {
#if defined(TEST_BUILD_NATIVE)
  // KNOWN ISSUE: native does not produce the full expected shift range.
  TEST_IGNORE_MESSAGE("Test_OT_int8_big_range fails on native");
#else
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
    TEST_ASSERT_TRUE(is_in);
  }
  for (int i = FINAL_SHR_RANGE_MIN; i <= FINAL_SHR_RANGE_MAX; i++) {
    const bool is_in = seen_final_shr.find(i) != seen_final_shr.end();
    TEST_ASSERT_TRUE(is_in);
  }
#endif
}

TEST(group_output_transforms, Test_OT_int8_channelwise_big_range) {
#if defined(TEST_BUILD_NATIVE)
  // KNOWN ISSUE: native does not produce the full expected shift range.
  TEST_IGNORE_MESSAGE("Test_OT_int8_channelwise_big_range fails on native");
#else
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
    TEST_ASSERT_TRUE(is_in);
  }
  for (int i = FINAL_SHR_RANGE_MIN; i <= FINAL_SHR_RANGE_MAX; i++) {
    const bool is_in = seen_final_shr.find(i) != seen_final_shr.end();
    TEST_ASSERT_TRUE(is_in);
  }
#endif
}

// Three channels of a real VNR conv layer whose largest channel forces B == 0 with the VX4 VLMUL
// shift. Rounding their biases with a fixed 2^-B correction cost a whole output LSB there.
static MulsAndBias low_bias_precision_channels() {
  MulsAndBias mb;
  mb.push_back(OutputTransformFn::ActivationParams(-489.400, 0.015873, 22768, 38833));
  mb.push_back(OutputTransformFn::ActivationParams(-1842.950, 0.069603, 24639, 28303));
  mb.push_back(OutputTransformFn::ActivationParams(-12824.039, 0.072502, 175113, 178631));
  return mb;
}

TEST(group_output_transforms, Test_OT_int8_bias_low_precision) {
  for (nn_vlmul_shr_t shr : {VLMUL_SHR_XS3A, VLMUL_SHR_VX4A}) {
    MulsAndBias mb = low_bias_precision_channels();
    auto qp = OutputTransformFnInt8_Group::Quantizer().quantise_activation(mb, shr, false);
    double error = OutputTransformFnInt8::get_quant_error(mb, qp, shr, true);
    TEST_ASSERT_TRUE_MESSAGE(error < 0.5, "group quantisation error too high");
  }
}

TEST(group_output_transforms, Test_OT_int8_channelwise_bias_low_precision) {
  for (nn_vlmul_shr_t shr : {VLMUL_SHR_XS3A, VLMUL_SHR_VX4A}) {
    MulsAndBias mb = low_bias_precision_channels();
    auto qp = OutputTransformFnInt8_Channelwise::Quantizer().quantise_activation(mb, shr, false);
    double error = OutputTransformFnInt8_Channelwise::get_quant_error(mb, qp, shr, true);
    TEST_ASSERT_TRUE_MESSAGE(error < 0.5, "channelwise quantisation error too high");
  }
}

// Mean signed error of the quantised group transform against the unrounded float result, over
// every accumulator of every channel. get_quant_error() is an absolute error, so it does not see
// a constant output offset that is smaller than the rounding steps.
static double group_mean_signed_error(MulsAndBias &mb,
                                      OutputTransformFnInt8_Group::QuantisationParams &qp,
                                      nn_vlmul_shr_t shr) {
  double error_sum = 0.0;
  int64_t count = 0;
  for (unsigned ch = 0; ch < mb.size(); ++ch) {
    for (int accu = mb[ch].accu_min_val; accu <= mb[ch].accu_max_val; ++accu) {
      int32_t t = OutputTransformFnInt8::shr(accu, qp.initial_shr);                // vlsat
      t = OutputTransformFnInt8::mul(t, qp.multipliers[ch], 16, shr);              // vlmul
      t = OutputTransformFnInt8::add(t, qp.biases[ch]);                            // vladd
      t = OutputTransformFnInt8::shr(t, qp.final_shr);                             // vlashr
      t = OutputTransformFnInt8::sat(OutputTransformFnInt8::shr(t, 8), 8);         // vdepth8
      error_sum += t - ((double)accu * mb[ch].multiplier + mb[ch].bias);
      count++;
    }
  }
  return error_sum / (double)count;
}

// Small accumulators make the group quantiser shift them left (initial_shr < 0), and a multiplier
// of 2.5 quantises to 5 * 2^12. Every VLMUL product is then a multiple of 2^(vlmul_shr - 1), so
// half of them round up by half an LSB. That rounding drift is 0.25 output LSB at B == 0, not the
// 2^-(vlmul_shr + 1) of uniformly spread products, and the bias has to remove it.
TEST(group_output_transforms, Test_OT_int8_bias_left_shifted_accu) {
  for (nn_vlmul_shr_t shr : {VLMUL_SHR_XS3A, VLMUL_SHR_VX4A}) {
    MulsAndBias mb;
    for (int ch = 0; ch < 16; ++ch) {
      int32_t accu_min = 4300 + 230 * ch;
      int32_t accu_max = accu_min + 80;
      // Spread the biases' fractional parts so that the channels round their biases differently
      double bias = -2.5 * 0.5 * (accu_min + accu_max) + (ch / 16.0 - 0.5) + 0.03;
      mb.push_back(OutputTransformFn::ActivationParams(bias, 2.5, accu_min, accu_max));
    }
    auto qp = OutputTransformFnInt8_Group::Quantizer().quantise_activation(mb, shr, false);
    TEST_ASSERT_TRUE_MESSAGE(qp.initial_shr < 0, "expected the accumulators to be shifted left");

    double offset = group_mean_signed_error(mb, qp, shr);
    TEST_ASSERT_TRUE_MESSAGE(std::fabs(offset) < 0.1, "group quantisation has an output offset");
    // Half of the channels have every other output on a rounding tie, so they are about 0.5 LSB
    // out whichever bias is chosen
    double error = OutputTransformFnInt8::get_quant_error(mb, qp, shr, true);
    TEST_ASSERT_TRUE_MESSAGE(error < 0.55, "group quantisation error too high");
  }
}

// int8 output of one channel of a quantised transform for one accumulator
static int32_t int8_output(int initial_shr, int16_t multiplier, int16_t bias, int final_shr,
                           int32_t accu, nn_vlmul_shr_t shr) {
  int32_t t = OutputTransformFnInt8::shr(accu, initial_shr);                // vlsat
  t = OutputTransformFnInt8::mul(t, multiplier, 16, shr);                   // vlmul
  t = OutputTransformFnInt8::add(t, bias);                                  // vladd
  t = OutputTransformFnInt8::shr(t, final_shr);                             // vlashr
  return OutputTransformFnInt8::sat(OutputTransformFnInt8::shr(t, 8), 8);   // vdepth8
}

// Zero-multiplier channels, with biases on rounding ties of both signs. Their output is the bias
// alone, so it must be exactly the reference's std::round(bias). With only these channels B == 0;
// adding a channel with a small multiplier raises B so the final shifts round too. The constructor
// zeroes the bias of a zero-multiplier channel, so the bias is set afterwards.
static MulsAndBias zero_multiplier_channels(bool with_product) {
  MulsAndBias mb;
  for (double bias : {1.5, 2.5, -1.5, -2.5, 0.5, -0.5, 3.0, 100.4, -127.5}) {
    mb.push_back(OutputTransformFn::ActivationParams(bias, 0.0, -100, 100));
    mb.back().bias = bias;
  }
  if (with_product)
    mb.push_back(OutputTransformFn::ActivationParams(0.0, 0.01, 0, 1000));
  return mb;
}

TEST(group_output_transforms, Test_OT_int8_zero_multiplier) {
  for (nn_vlmul_shr_t shr : {VLMUL_SHR_XS3A, VLMUL_SHR_VX4A}) {
    for (bool with_product : {false, true}) {
      MulsAndBias mb = zero_multiplier_channels(with_product);
      auto qg = OutputTransformFnInt8_Group::Quantizer().quantise_activation(mb, shr, false);
      auto qc = OutputTransformFnInt8_Channelwise::Quantizer().quantise_activation(mb, shr, false);
      TEST_ASSERT_TRUE_MESSAGE((qg.final_shr > -8) == with_product, "unexpected group B");
      for (unsigned ch = 0; ch < mb.size(); ++ch) {
        if (mb[ch].multiplier != 0.0) continue;
        int expected = (int)std::round(mb[ch].bias);
        const int32_t accus[] = {mb[ch].accu_min_val, 0, mb[ch].accu_max_val};
        for (int32_t accu : accus) {
          TEST_ASSERT_EQUAL_INT32_MESSAGE(
              expected,
              int8_output(qg.initial_shr, qg.multipliers[ch], qg.biases[ch], qg.final_shr, accu, shr),
              "group zero-multiplier output");
          TEST_ASSERT_EQUAL_INT32_MESSAGE(
              expected,
              int8_output(qc.initial_shifts[ch], qc.multipliers[ch], qc.biases[ch], qc.final_shr,
                          accu, shr),
              "channelwise zero-multiplier output");
        }
      }
    }
  }
}

// A channel with only ten accumulators, grouped with a channel whose multiplier of 3 limits M to
// 13. The narrow channel then quantises to initial_shr 1, multiplier -21 and B -2 (XS3), so every
// product rounds the same way and the uniform rounding drift misplaces its bias by about half a
// bias LSB: 2 output LSBs at B == -2.
TEST(group_output_transforms, Test_OT_int8_bias_narrow_range) {
  for (nn_vlmul_shr_t shr : {VLMUL_SHR_XS3A, VLMUL_SHR_VX4A}) {
    MulsAndBias mb;
    mb.push_back(OutputTransformFn::ActivationParams(189.44601749396065, -0.0026232995180676013,
                                                     47523, 47532));
    mb.push_back(OutputTransformFn::ActivationParams(0.0, 3.0, -40, 40));
    auto qp = OutputTransformFnInt8_Group::Quantizer().quantise_activation(mb, shr, false);
    // B == final_shr + 8
    TEST_ASSERT_TRUE_MESSAGE(qp.final_shr < -8, "expected a negative bias exponent");

    double abs_error_sum = 0.0;
    for (int32_t accu = mb[0].accu_min_val; accu <= mb[0].accu_max_val; ++accu) {
      int expected = (int)std::round((double)accu * mb[0].multiplier + mb[0].bias);
      abs_error_sum += std::abs(expected - int8_output(qp.initial_shr, qp.multipliers[0],
                                                       qp.biases[0], qp.final_shr, accu, shr));
    }
    double error = abs_error_sum / (mb[0].accu_max_val - mb[0].accu_min_val + 1);
    // With B == -2 the outputs step by 4, so the best reachable error here is about 1 LSB
    TEST_ASSERT_TRUE_MESSAGE(error < 1.5, "narrow-range channel quantisation error too high");
  }
}

}  // extern "C"
