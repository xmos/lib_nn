#include <algorithm>
#include <cmath>
#include <random>

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

void test_big_range(int coef_count, int N) {
  int64_t bias_low = -(8LL << N);
  int64_t bias_high = (8LL << N) - 1;

  double p_low = -(1LL << std::max(N - 16, 0));
  double p_high = (1LL << N) - 1;

  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  for (int output_ch_count = 4; output_ch_count <= 64; output_ch_count += 4) {
    for (int itt = 0; itt < 32; itt++) {
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
            product_target / (float)std::max(-accu_min, accu_max);

        // the bias must be between [-2**(N+1) - 1, 2**(N+1)]
        double bias =
            (double)get_random_uniform_from_range(bias_low, bias_high);

        OutputTransformFn::ActivationParams a(bias, multiplier, accu_min,
                                              accu_max);
        mul_and_biases.push_back(a);
      }

      QuantisationParams qp =
          OutputTransformFnInt8::quantise_activation(mul_and_biases);

      // pad q.multipliers_and_biases to a multiple of VPU_INT16_EPV
      // this is to work around array over reads - padding wont effect the
      // result.
      int16_t pad_val = rng.rand<int16_t>();

      OutputTransformFn::pad_final_access(qp.multipliers_and_biases,
                                          VPU_INT16_EPV, pad_val);

      OT_int8::Params p((int32_t)output_ch_count, qp.initial_shr, qp.final_shr);

      OT_int8 ot(&p);
      ot.setMultipliersAndBiases(qp.multipliers_and_biases.data());

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

          next_y = ot.output_transform_fn(y, &A, ocg);

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int actual_output_channel =
                output_chan + ocg * vpu_ring_buffer_length;

            double expected =
                (double)accu_values[output_chan] *
                    mul_and_biases[actual_output_channel].original_multiplier +
                mul_and_biases[actual_output_channel].original_bias;

            expected = std::round(std::min(std::max(expected, (double)INT8_MIN),
                                           (double)INT8_MAX));

            int actual = (int)Y[actual_output_channel];

            // if(std::abs((int)expected  - actual) > 1){

            //   std::cout << "Error, ch:" << actual_output_channel<<std::endl;

            //   for(int ch=0;ch < mul_and_biases.size();ch++){
            //     std::cout << ch << " accu_min: " <<
            //     mul_and_biases[ch].original_accu_min_val << " accu_max:"
            //     << mul_and_biases[ch].original_accu_max_val << " bias:"
            //     << mul_and_biases[ch].original_bias << " mul:"
            //     << mul_and_biases[ch].original_multiplier << std::endl;
            //   }
            //   std::cout << "output_chan " << output_chan <<std::endl;
            //   std::cout << "actual_output_channel " << actual_output_channel
            //   <<std::endl; std::cout << "accu_values[output_chan] " <<
            //   accu_values[output_chan] <<std::endl; std::cout << "expected "
            //   << expected <<std::endl; std::cout << "actual " << actual
            //   <<std::endl;

            // }
            TEST_ASSERT_INT32_WITHIN(1, (int)expected, actual);
          }
        }
        y = next_y;
      }
    }
  }
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

    QuantisationParams qp =
        OutputTransformFnInt8::quantise_activation(mul_and_biases);

    // pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
    // this is to work around array over reads
    int16_t pad_val = rng.rand<int16_t>();  // this is arbitrary

    OutputTransformFn::pad_final_access(qp.multipliers_and_biases,
                                        VPU_INT16_EPV, pad_val);

    OT_int8::Params p((int32_t)output_ch_count, qp.initial_shr, qp.final_shr);

    OT_int8 ot(&p);
    ot.setMultipliersAndBiases(qp.multipliers_and_biases.data());

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

        next_y = ot.output_transform_fn(y, &A, ocg);

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

void Test_OT_int8_range_small_bias() {
  test_small_range(INT8_MIN, INT8_MAX, 1);
}

void Test_OT_int8_range_small_bias2() {
  test_small_range(INT8_MIN, INT8_MAX, -1);
}

void Test_OT_int8_small_range() {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, 0);
}

void Test_OT_int8_small_range_bias() {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, 1);
}
void Test_OT_int8_small_range_bias2() {
  test_small_range(INT8_MIN - 10, INT8_MAX + 10, -1);
}

void Test_OT_int8_small_range_wide_bias_range() {
  for (int bias = INT8_MIN * 2 - 10; bias < 48; bias++) {
    test_small_range(INT8_MIN, INT8_MAX, bias);
  }
}

void Test_OT_int8_small_range_massive_bias_range() {
  for (int itt = 0; itt < 32; ++itt) {
    int32_t bias = rng.rand<int32_t>(1 << 15, 1 << 30);
    test_small_range(INT8_MIN, INT8_MAX, bias);
  }
}

void Test_OT_int8_big_range() {
  for (int n = 6; n < 12; n++) {
    for (int coef_count_log2 = 3; coef_count_log2 < 12; coef_count_log2++) {
      test_big_range(1 << coef_count_log2, n);
    }
  }
}

extern "C" void test_output_transforms();
void test_output_transforms() {
  UNITY_SET_FILE();
  RUN_TEST(Test_OT_int8_range);
  RUN_TEST(Test_OT_int8_range_small_bias);
  RUN_TEST(Test_OT_int8_range_small_bias2);
  RUN_TEST(Test_OT_int8_small_range);
  RUN_TEST(Test_OT_int8_small_range_bias);
  RUN_TEST(Test_OT_int8_small_range_bias2);
  RUN_TEST(Test_OT_int8_small_range_massive_bias_range);
  RUN_TEST(Test_OT_int8_small_range_wide_bias_range);
  RUN_TEST(Test_OT_int8_big_range);
}
