#include <cmath>
#include <random>

#include "OutputTransformFn.hpp"
#include "Rand.hpp"
#include "gtest/gtest.h"

namespace nn {
static auto rng = test::Rand(42);

void pick_activation_params(std::vector<double> &multiplier,
                            std::vector<double> &bias,
                            std::vector<int32_t> &accu_min,
                            std::vector<int32_t> &accu_max) {
  int multiplier_dynamic_range = 7;

  int p = 9;
  for (int ch = 0; ch < multiplier.size(); ch++) {
    int32_t abs_max_accu =
        std::max(std::abs(accu_min[ch]), std::abs(accu_max[ch]));

    double r =
        std::pow(2.0, std::abs(rng.rand<double>()) * multiplier_dynamic_range) /
        std::pow(2.0, multiplier_dynamic_range);
    double s = rng.rand<double>();
    multiplier[ch] = std::pow(2.0, p) / abs_max_accu * r * s / std::abs(s);

    double prod_min = multiplier[ch] * accu_min[ch];
    double prod_max = multiplier[ch] * accu_max[ch];

    bias[ch] = prod_min + std::abs(rng.rand<double>()) * (prod_max - prod_min);
  }
}

void pick_accu_range(std::vector<int32_t> &accu_min,
                     std::vector<int32_t> &accu_max) {
  int coef_count = (int)((rng.rand<uint32_t>() % (255 - 32)) + 32);

  // std::random_device rd;
  // std::mt19937 gen(rd());
  // std::normal_distribution<> d{5, 2};

  for (int ch = 0; ch < accu_min.size(); ch++) {
    accu_min[ch] = 0;
    accu_max[ch] = 0;
    for (int i = 0; i < coef_count; i++) {
      int8_t b = rng.rand<int8_t>();
      while (b == 0) b = rng.rand<int8_t>();

      if (b > 0) {
        accu_min[ch] += (int32_t)b * (int32_t)INT8_MIN;
        accu_max[ch] += (int32_t)b * (int32_t)INT8_MAX;
      } else {
        accu_min[ch] += (int32_t)b * (int32_t)INT8_MAX;
        accu_max[ch] += (int32_t)b * (int32_t)INT8_MIN;
      }
    }
  }

  int32_t min_it = 0;
  int32_t max_it = 0;
  for (int32_t a : accu_min) {
    min_it = std::min(min_it, a);
    max_it = std::max(max_it, a);
  }

  assert(std::abs((double)max_it / min_it) < 2);
}

class Test_OT_int8 : public ::testing::Test {};

TEST_F(Test_OT_int8, BasicTest) {
  const int vpu_ring_buffer_length = VPU_INT16_EPV;

  for (int output_ch_count = 1; output_ch_count <= 64; ++output_ch_count) {
    for (int itt = 0; itt < 1 << 8; itt++) {
      std::vector<double> f_biases(output_ch_count, 0);
      std::vector<double> f_multipliers(output_ch_count, 0);
      std::vector<int32_t> accu_min(output_ch_count, 0);
      std::vector<int32_t> accu_max(output_ch_count, 0);

      pick_accu_range(accu_min, accu_max);

      pick_activation_params(f_multipliers, f_biases, accu_max, accu_min);

      QuantisationParams qp = OutputTransformFnInt8::quantise_activation(
          f_multipliers, f_biases, accu_min, accu_max);

      OT_int8::Params p((int32_t)output_ch_count, &qp.otv, qp.biases.data(),
                        qp.multipliers.data());

      OT_int8 ot(&p);

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
        for (int t = 0; t < 1 << 12; t++) {
          memset(&A, 0, sizeof A);

          int32_t accu_values[chs_in_group];
          // fill A with random value between accu_max and accu_min
          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int64_t range =
                (int64_t)accu_max[output_chan] - (int64_t)accu_min[output_chan];
            ASSERT_NE(0, range) << "Test case attempted division by zero.";
            int32_t v =
                (int64_t)accu_min[output_chan] + (rng.rand<unsigned>()) % range;

            accu_values[output_chan] = v;
            A.vR[output_chan] = ((int16_t *)&v)[0];
            A.vD[output_chan] = ((int16_t *)&v)[1];
          }

          next_y = ot.output_transform_fn(y, &A, ocg);

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
            int actual_output_channel =
                output_chan + ocg * vpu_ring_buffer_length;

            int32_t v = accu_values[output_chan];
            double expected = (float)v * f_multipliers[actual_output_channel] +
                              f_biases[actual_output_channel];
            double f_expected = std::round(std::min(
                std::max(expected, (double)INT8_MIN), (double)INT8_MAX));
            EXPECT_NEAR((int)f_expected, (int)Y[actual_output_channel], 1)
                << expected
                << " actual_output_channel: " << actual_output_channel
                << " output_ch_count: " << output_ch_count
                << " accu_value: " << v
                << " f_biases: " << f_biases[actual_output_channel]
                << " f_multipliers: " << f_multipliers[actual_output_channel]
                << " accu_max: " << accu_max[actual_output_channel]
                << " accu_min: " << accu_min[actual_output_channel];
          }
        }
        y = next_y;
      }
    }
  }
}
}  // namespace nn
