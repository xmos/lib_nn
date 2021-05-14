#include "gtest/gtest.h"

#include <cmath>
#include "Rand.hpp"
#include "OutputTransformFn.hpp"

namespace nn
{
  static auto rng = test::Rand(42);

  std::pair<double, double> pick_mul_and_bias(double output_min, double output_max, double accu_min, double accu_max, int *seed)
  {
    double output_overscale = 1.1 + 0.2 * (double)rng.rand<int32_t>() / (double)INT32_MAX;
    double output_range = (output_max - output_min) * output_overscale;
    double mul = output_range / (double)(accu_max - (double)accu_min);
    double bias = output_min * output_overscale - accu_min * mul;
    return std::make_pair(mul, bias);
  }

  void pick_activation_params(
      std::vector<double> &multiplier,
      std::vector<double> &bias,
      std::vector<int32_t> &accu_min,
      std::vector<int32_t> &accu_max,
      int *seed)
  {

    double output_min = (double)INT8_MIN;
    double output_max = (double)INT8_MAX;

    std::tie(multiplier[0], bias[0]) = pick_mul_and_bias(output_min, output_max, accu_min[0], accu_max[0], seed);

    double max_abs_mul = std::abs(multiplier[0]);
    double min_abs_mul = max_abs_mul;

    for (int ch = 1; ch < multiplier.size(); ch++)
    {

      std::tie(multiplier[ch], bias[ch]) = pick_mul_and_bias(output_min, output_max, accu_min[ch], accu_max[ch], seed);

      while (std::max(max_abs_mul, std::abs(multiplier[ch])) / std::min(min_abs_mul, std::abs(multiplier[ch])) > 63)
      {
        std::tie(multiplier[ch], bias[ch]) = pick_mul_and_bias(output_min, output_max, accu_min[ch], accu_max[ch], seed);
      }
      max_abs_mul = std::max(max_abs_mul, std::abs(multiplier[ch]));
      min_abs_mul = std::min(min_abs_mul, std::abs(multiplier[ch]));
    }
  }

  void pick_accu_range(std::vector<int32_t> &accu_min, std::vector<int32_t> &accu_max, int *seed)
  {

    //It's reasonable to assume all accumulators are approximatly of the same order
    int scale = rng.rand<int32_t>() % 24;
    if (scale < 0)
      scale = -scale;
    scale += 1;

    int32_t a = 0;
    while (!a)
      a = rng.rand<int32_t>() >> scale;

    for (int ch = 0; ch < accu_min.size(); ch++)
    {

      int32_t b = ((rng.rand<int32_t>() % a) >> (scale + 2)) - a;

      accu_max[ch] = std::max(a, b);
      accu_min[ch] = std::min(a, b);
    }
  }

  // void pick_accu_big_range(std::vector<int32_t> &accu_min, std::vector<int32_t> &accu_max, int * seed){

  //   //It's reasonable to assume all accumulators are approximatly of the same order
  //   int scale =  pseudo_rand(seed)%24;
  //   if(scale < 0)
  //     scale = -scale;
  //   scale += 1;

  //   for (unsigned ch = 0; ch < accu_min.size(); ch++){
  //     int32_t a=0, b=0;
  //     while (!a) a = pseudo_rand(seed)>>scale;
  //     while (!b) b = pseudo_rand(seed)>>scale;

  //     accu_max[ch] = std::max(a, b);
  //     accu_min[ch] = std::min(a, b);
  //   }
  // }

  class Test_OT_int8 : public ::testing::Test
  {
  };

  TEST_F(Test_OT_int8, BasicTest)
  {

    const int vpu_ring_buffer_length = VPU_INT16_EPV;

    int seed = 69;

    for (int output_ch_count = 2; output_ch_count <= 64; ++output_ch_count)
    {

      for (int itt = 0; itt < 1 << 8; itt++)
      {

        std::vector<double> f_biases(output_ch_count, 0);
        std::vector<double> f_multipliers(output_ch_count, 0);
        std::vector<int32_t> accu_min(output_ch_count, 0);
        std::vector<int32_t> accu_max(output_ch_count, 0);

        pick_accu_range(accu_min, accu_max, &seed);

        pick_activation_params(f_multipliers, f_biases, accu_max, accu_min, &seed);

        QuantisationParams qp = OutputTransformFnInt8::quantise_activation(f_multipliers, f_biases, accu_min, accu_max);

        OT_int8::Params p((int32_t)output_ch_count, &qp.otv, qp.biases.data(),
                          qp.multipliers.data());

        OT_int8 ot(&p);

        int8_t Y[output_ch_count];
        memset(Y, 0, sizeof Y);

        int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

        int8_t *y = (int8_t *)Y;

        for (int ocg = 0; ocg < ocg_count; ++ocg)
        {

          int chs_in_group = std::min(output_ch_count - vpu_ring_buffer_length * ocg, vpu_ring_buffer_length);

          vpu_ring_buffer_t A;

          int8_t *next_y;
          for (int t = 0; t < 1 << 8; t++)
          {

            memset(&A, 0, sizeof A);

            int32_t accu_values[chs_in_group];
            //fill A with random value between accu_max and accu_min
            for (int output_chan = 0; output_chan < chs_in_group; ++output_chan)
            {

              int64_t range = (int64_t)accu_max[output_chan] - (int64_t)accu_min[output_chan];
              ASSERT_NE(0, range) << "Test case attempted division by zero.";
              int32_t v = (int64_t)accu_min[output_chan] + (rng.rand<unsigned>()) % range;

              accu_values[output_chan] = v;
              A.vR[output_chan] = ((int16_t *)&v)[0];
              A.vD[output_chan] = ((int16_t *)&v)[1];
            }

            next_y = ot.output_transform_fn(y, &A, ocg);

            for (int output_chan = 0; output_chan < chs_in_group; ++output_chan)
            {

              int actual_output_channel = output_chan + ocg * vpu_ring_buffer_length;

              int32_t v = accu_values[output_chan];
              double expected = (float)v * f_multipliers[actual_output_channel] + f_biases[actual_output_channel];
              expected = std::round(std::min(std::max(expected, (double)INT8_MIN), (double)INT8_MAX));
              EXPECT_NEAR((int)expected, (int)Y[actual_output_channel], 1);
            }
          }
          y = next_y;
        }
      }
    }
  }

}
