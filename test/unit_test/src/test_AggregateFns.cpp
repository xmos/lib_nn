// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <list>
#include <tuple>
#include <vector>

#include "AggregateFn.hpp"
#include "Rand.hpp"

extern "C" {
#include "expand_8_to_16.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"
}
using namespace nn;
using namespace nn::test;

static auto rng = test::Rand(42);

extern "C" {

TEST_GROUP(group_aggregate_fns);
TEST_SETUP(group_aggregate_fns) {}
TEST_TEAR_DOWN(group_aggregate_fns) {}
TEST_GROUP_RUNNER(group_aggregate_fns) {
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulDirectFn_int16_DW);
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulDirectFn_int16);
  RUN_TEST_CASE(group_aggregate_fns, Test_SimpleMatMulInt8);
  RUN_TEST_CASE(group_aggregate_fns, Test_SimpleMatMulBinary);
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulInt8);
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulBinary);
  RUN_TEST_CASE(group_aggregate_fns, Test_Simple_MatMulDirectFn);
  RUN_TEST_CASE(group_aggregate_fns, Test_Simple_MatMulBinaryDirectFn);
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulDirectFn);
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulBinaryDirectFn);
  RUN_TEST_CASE(group_aggregate_fns, Test_Kernel_Reordering);
  RUN_TEST_CASE(group_aggregate_fns, Test_Simple_MatMulDirectFn_DW);
  RUN_TEST_CASE(group_aggregate_fns, Test_MatMulDirectFn_DW);
  RUN_TEST_CASE(group_aggregate_fns, Test_Kernel_Reordering_DW);
}

/*
  Simple test to verify memory accesses
*/
TEST(group_aggregate_fns, Test_SimpleMatMulInt8) {
  const int vpu_ring_buffer_length = 16;

  for (auto input_bytes = 4; input_bytes < 48; input_bytes += 4) {
    std::list<std::tuple<int8_t, int8_t> > args = {
        std::tuple<int8_t, int8_t>{1, 1},  std::tuple<int8_t, int8_t>{1, 0},
        std::tuple<int8_t, int8_t>{0, 1},  std::tuple<int8_t, int8_t>{-1, 1},
        std::tuple<int8_t, int8_t>{1, -1}, std::tuple<int8_t, int8_t>{-1, -1},
    };

    for (auto arg : args) {
      int8_t kernel_fill, scratch_fill;
      std::tie(kernel_fill, scratch_fill) = arg;

      for (int output_channel_count = 1; output_channel_count < 48;
           ++output_channel_count) {
        int scratch_bytes = MatMulInt8::get_scratch_mem_bytes(input_bytes);
        int kernel_bytes =
            MatMulInt8::get_weights_bytes(input_bytes, output_channel_count);

        std::vector<int8_t> K(kernel_bytes);
        std::vector<int8_t> T(scratch_bytes);

        MatMulInt8 mm(output_channel_count, input_bytes);
        mat_mul_generic_params_t p = mm.getParams();

        std::fill_n(K.data(), kernel_bytes, kernel_fill);
        std::fill_n(T.data(), scratch_bytes, scratch_fill);

        int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) /
                        vpu_ring_buffer_length;

        for (int ocg = 0; ocg < ocg_count; ++ocg) {
          alignas(4) VPURingBuffer A;
          mat_mul_generic_int8(&p, &A, T.data(), ocg, K.data());

          int c;
          if ((ocg + 1) * vpu_ring_buffer_length < output_channel_count)
            c = vpu_ring_buffer_length;
          else
            c = output_channel_count % vpu_ring_buffer_length;

          for (int output_chan = 0; output_chan < c; ++output_chan) {
            int32_t v;
            ((int16_t *)&v)[0] = A.vR[output_chan];
            ((int16_t *)&v)[1] = A.vD[output_chan];

            TEST_ASSERT_EQUAL(scratch_bytes * (kernel_fill * scratch_fill), v);
          }
        }
      }
    }
  }
}

void accumulate_binary_bytes(int *accu, int8_t a, int8_t b) {
  int t = (a ^ b);
  *accu += ((2 * __builtin_popcount((~t) & 0xff) - CHAR_BIT) / 2);
}
/*
  Simple test to verify memory accesses
*/
TEST(group_aggregate_fns, Test_SimpleMatMulBinary) {
  const int vpu_ring_buffer_length = 16;

  for (auto input_bytes = 4; input_bytes < 48; input_bytes += 4) {
    std::list<std::tuple<int8_t, int8_t> > args = {
        std::tuple<int8_t, int8_t>{-1, -1},
        std::tuple<int8_t, int8_t>{-1, 0},
        std::tuple<int8_t, int8_t>{0, 0},
        std::tuple<int8_t, int8_t>{0, -1},
    };

    for (auto arg : args) {
      int8_t kernel_fill, scratch_fill;
      std::tie(kernel_fill, scratch_fill) = arg;

      for (int output_channel_count = 8; output_channel_count < 48;
           output_channel_count += 8) {
        int scratch_bytes = MatMulInt8::get_scratch_mem_bytes(input_bytes);
        int kernel_bytes =
            MatMulInt8::get_weights_bytes(input_bytes, output_channel_count);

        std::vector<int8_t> K(kernel_bytes);
        std::vector<int8_t> T(scratch_bytes);

        MatMulBinary mm(output_channel_count, input_bytes);
        mat_mul_generic_params_t p = mm.getParams();

        std::fill_n(K.data(), kernel_bytes, kernel_fill);
        std::fill_n(T.data(), scratch_bytes, scratch_fill);

        int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) /
                        vpu_ring_buffer_length;

        for (int ocg = 0; ocg < ocg_count; ++ocg) {
          alignas(4) VPURingBuffer A;
          mat_mul_generic_binary(&p, &A, T.data(), ocg, K.data());

          int c;
          if ((ocg + 1) * vpu_ring_buffer_length < output_channel_count)
            c = vpu_ring_buffer_length;
          else
            c = output_channel_count % vpu_ring_buffer_length;

          int expected = 0;
          accumulate_binary_bytes(&expected, kernel_fill, scratch_fill);
          expected *= scratch_bytes;

          for (int output_chan = 0; output_chan < c; ++output_chan) {
            int32_t v;
            ((int16_t *)&v)[0] = A.vR[output_chan];
            ((int16_t *)&v)[1] = A.vD[output_chan];

            TEST_ASSERT_EQUAL(expected, v);
          }
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses
*/
TEST(group_aggregate_fns, Test_MatMulInt8) {
  const int vpu_bytes = 32;
  const int vpu_ring_buffer_length = 16;

  for (int input_bytes = 4; input_bytes < 128; input_bytes += 4) {
    for (int output_channel_count = 1; output_channel_count < 48;
         ++output_channel_count) {
      int k_height = 1;
      int k_width = 1;  // to make things easy

      std::array<int, 4> shape = {{output_channel_count, k_height, k_width, input_bytes}};
      // k_height == k_width == 1, so raw_weights[oc][0][0][b] == raw_weights[oc * input_bytes + b]
      std::vector<int8_t> raw_weights(output_channel_count * k_height * k_width * input_bytes);

      for (size_t j = 0; j < raw_weights.size(); ++j)
        raw_weights[j] = rng.rand<int8_t>();

      int scratch_bytes = MatMulInt8::get_scratch_mem_bytes(input_bytes);

      int8_t pad_val = rng.rand<int8_t>();

      Conv2dReorderedWeights rw = MatMulInt8::reorder_kernel_weights(
          raw_weights.data(), shape, 8, pad_val);

      std::vector<int8_t> T(scratch_bytes);

      for (int j = 0; j < scratch_bytes; ++j) T[j] = rng.rand<int8_t>();

      std::vector<int> accu_modifier(output_channel_count);  //=0

      // TODO make this into an int8 specific function
      for (int i = 0; i < output_channel_count; i++) {
        int idx = rw.final_vpu_load_addresses[i];

        int s = 0;
        int channel_overlap_start = input_bytes % vpu_bytes;

        if (channel_overlap_start) {
          for (int j = channel_overlap_start; j < vpu_bytes; j++) {
            s += (int)(rw.weights[idx + j]) * T[scratch_bytes - vpu_bytes + j];
          }
        }
        accu_modifier[i] = s;
      }

      std::vector<int8_t> reordered_weights(rw.weights.size());
      std::memcpy(reordered_weights.data(), rw.weights.data(), rw.weights.size());

      MatMulInt8 mm(output_channel_count,
                           input_bytes);  // reordered_weights
      mat_mul_generic_params_t p = mm.getParams();

      int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) /
                      vpu_ring_buffer_length;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        alignas(4) VPURingBuffer A;
        mat_mul_generic_int8(&p, &A, T.data(), ocg, reordered_weights.data());

        int chs_in_group =
            std::min(output_channel_count - output_channel_count * ocg,
                     vpu_ring_buffer_length);

        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int actual_output_channel =
              output_chan + ocg * vpu_ring_buffer_length;

          int expected_sum = 0;
          for (int b = 0; b < input_bytes; b++)
            expected_sum +=
                ((int)raw_weights[actual_output_channel * input_bytes + b] * (int)T[b]);

          int32_t v;
          ((int16_t *)&v)[0] = A.vR[output_chan];
          ((int16_t *)&v)[1] = A.vD[output_chan];

          TEST_ASSERT_EQUAL(v - accu_modifier[actual_output_channel],
                            expected_sum);
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses
*/
TEST(group_aggregate_fns, Test_MatMulBinary) {
  const int vpu_bytes = 32;
  const int vpu_ring_buffer_length = 16;

  for (int input_bytes = 4; input_bytes < 128; input_bytes += 4) {
    for (int output_channel_count = 8; output_channel_count < 48;
         output_channel_count += 8) {
      int k_height = 1;
      int k_width = 1;  // to make things easy

      std::array<int, 4> shape = {
          {output_channel_count, k_height, k_width, input_bytes}};
      // k_height == k_width == 1, so raw_weights[oc][0][0][b] == raw_weights[oc * input_bytes + b]
      std::vector<int8_t> raw_weights(output_channel_count * k_height * k_width * input_bytes);

      for (size_t j = 0; j < raw_weights.size(); ++j)
        raw_weights[j] = rng.rand<int8_t>();

      int scratch_bytes = MatMulInt8::get_scratch_mem_bytes(input_bytes);

      int8_t pad_val = rng.rand<int8_t>();

      Conv2dReorderedWeights rw = MatMulInt8::reorder_kernel_weights(
          raw_weights.data(), shape, 8, pad_val);

      std::vector<int8_t> T(scratch_bytes);

      for (int j = 0; j < scratch_bytes; ++j) T[j] = rng.rand<int8_t>();

      std::vector<int> expected(output_channel_count);
      for (int i = 0; i < output_channel_count; i++) {
        expected[i] = 0;
        for (int j = 0; j < scratch_bytes; ++j) {
          accumulate_binary_bytes(&(expected[i]), raw_weights[j], T[j]);
        }
      }

      std::vector<int> accu_modifier(output_channel_count);  //=0

      // TODO make this into an int8 specific function
      for (int i = 0; i < output_channel_count; i++) {
        int idx = rw.final_vpu_load_addresses[i];

        int s = 0;
        int channel_overlap_start = input_bytes % vpu_bytes;

        if (channel_overlap_start) {
          for (int j = channel_overlap_start; j < vpu_bytes; j++) {
            s += (int)(rw.weights[idx + j]) * T[scratch_bytes - vpu_bytes + j];
          }
        }
        accu_modifier[i] = s;
      }

      std::vector<int8_t> reordered_weights(rw.weights.size());
      std::memcpy(reordered_weights.data(), rw.weights.data(), rw.weights.size());

      MatMulInt8 mm(output_channel_count,
                           input_bytes);  // reordered_weights
      mat_mul_generic_params_t p = mm.getParams();
      int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) /
                      vpu_ring_buffer_length;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        alignas(4) VPURingBuffer A;
        mat_mul_generic_int8(&p, &A, T.data(), ocg, reordered_weights.data());

        int chs_in_group =
            std::min(output_channel_count - output_channel_count * ocg,
                     vpu_ring_buffer_length);

        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int actual_output_channel =
              output_chan + ocg * vpu_ring_buffer_length;

          int expected_sum = 0;
          for (int b = 0; b < input_bytes; b++)
            expected_sum +=
                ((int)raw_weights[actual_output_channel * input_bytes + b] * (int)T[b]);

          int32_t v;
          ((int16_t *)&v)[0] = A.vR[output_chan];
          ((int16_t *)&v)[1] = A.vD[output_chan];

          TEST_ASSERT_EQUAL(v - accu_modifier[actual_output_channel],
                            expected_sum);
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses.
*/
TEST(group_aggregate_fns, Test_Simple_MatMulDirectFn) {
  const int vpu_ring_buffer_length = 16;

  std::list<std::tuple<int8_t, int8_t> > args = {
      std::tuple<int8_t, int8_t>{1, 1},  std::tuple<int8_t, int8_t>{1, 0},
      std::tuple<int8_t, int8_t>{0, 1},  std::tuple<int8_t, int8_t>{-1, 1},
      std::tuple<int8_t, int8_t>{1, -1}, std::tuple<int8_t, int8_t>{-1, -1},
  };

  for (auto arg : args) {
    int8_t kernel_fill, scratch_fill;
    std::tie(kernel_fill, scratch_fill) = arg;

    for (int x_height = 1; x_height <= 4; ++x_height) {
      for (int x_width = 1; x_width <= 4; ++x_width) {
        for (int x_channels = 32; x_channels <= 32 * 3; x_channels += 32) {
          for (int k_height = 1; k_height <= x_height; ++k_height) {
            for (int k_width = 1; k_width <= x_width; ++k_width) {
              for (int y_channels = 32; y_channels < 32 * 3; y_channels += 32) {
                ImageGeometry X_params(x_height, x_width, x_channels);
                WindowGeometry K_params(k_height, k_width, 1, 1, 1, 1);

                std::vector<int8_t> K(y_channels * k_height * k_width * x_channels);
                std::vector<int8_t> T(x_height * x_width * x_channels);

                int8_t *weights =
                    K.data();  // todo we will switch to usnig the boggler

                MatMulDirectFn mmd(
                    X_params, K_params,
                    x_channels);  // weights, (int)(y_channels * k_height *
                                  // k_width * x_channels)
                mat_mul_direct_params_t p = mmd.getParams();

                std::fill_n(K.data(), K.size(), kernel_fill);
                std::fill_n(T.data(), T.size(), scratch_fill);

                int ocg_count = (y_channels + vpu_ring_buffer_length - 1) /
                                vpu_ring_buffer_length;

                for (int x = 0; x < x_height - k_height + 1; ++x) {
                  for (int y = 0; y < x_width - k_width + 1; ++y) {
                    for (int ocg = 0; ocg < ocg_count; ++ocg) {
                      alignas(4) VPURingBuffer A;
                      mat_mul_direct_int8(&p, &A, T.data(), ocg, weights);

                      for (int output_chan = 0;
                           output_chan < vpu_ring_buffer_length;
                           ++output_chan) {
                        int32_t v;
                        ((int16_t *)&v)[0] = A.vR[output_chan];
                        ((int16_t *)&v)[1] = A.vD[output_chan];

                        TEST_ASSERT_EQUAL(k_width * k_height * x_channels *
                                              (kernel_fill * scratch_fill),
                                          v);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/*
  Simple test to verify memory accesses.
*/
TEST(group_aggregate_fns, Test_Simple_MatMulBinaryDirectFn) {
  const int vpu_ring_buffer_length = 16;

  std::list<std::tuple<int8_t, int8_t> > args = {
      std::tuple<int8_t, int8_t>{-1, -1},
      std::tuple<int8_t, int8_t>{-1, 0},
      std::tuple<int8_t, int8_t>{0, 0},
      std::tuple<int8_t, int8_t>{0, -1},
  };

  for (auto arg : args) {
    int8_t kernel_fill, scratch_fill;
    std::tie(kernel_fill, scratch_fill) = arg;

    for (int x_height = 1; x_height <= 4; ++x_height) {
      for (int x_width = 1; x_width <= 4; ++x_width) {
        for (int x_channels = 256; x_channels <= 256 * 3; x_channels += 256) {
          for (int k_height = 1; k_height <= x_height; ++k_height) {
            for (int k_width = 1; k_width <= x_width; ++k_width) {
              for (int y_channels = 256; y_channels < 256 * 3;
                   y_channels += 256) {
                ImageGeometry X_params(x_height, x_width, x_channels, 1);
                WindowGeometry K_params(k_height, k_width, 1, 1, 1, 1);

                std::vector<int8_t> K(y_channels * k_height * k_width * x_channels / 8);
                std::vector<int8_t> T(x_height * x_width * x_channels / 8);

                int8_t *weights =
                    K.data();  // todo we will switch to usnig the boggler

                MatMulBinaryDirectFn mmd(X_params, K_params, x_channels);
                mat_mul_direct_params_t p = mmd.getParams();

                std::fill_n(K.data(), K.size(), kernel_fill);
                std::fill_n(T.data(), T.size(), scratch_fill);

                int expected = 0;
                accumulate_binary_bytes(&(expected), kernel_fill, scratch_fill);
                expected *= (k_height * k_width * x_channels / CHAR_BIT);

                int ocg_count = (y_channels + vpu_ring_buffer_length - 1) /
                                vpu_ring_buffer_length;

                for (int x = 0; x < x_height - k_height + 1; ++x) {
                  for (int y = 0; y < x_width - k_width + 1; ++y) {
                    for (int ocg = 0; ocg < ocg_count; ++ocg) {
                      alignas(4) VPURingBuffer A;
                      mat_mul_direct_binary(&p, &A, T.data(), ocg, weights);

                      for (int output_chan = 0;
                           output_chan < vpu_ring_buffer_length;
                           ++output_chan) {
                        int32_t v;
                        ((int16_t *)&v)[0] = A.vR[output_chan];
                        ((int16_t *)&v)[1] = A.vD[output_chan];

                        TEST_ASSERT_EQUAL(expected, v);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses.
*/
TEST(group_aggregate_fns, Test_MatMulDirectFn) {
  const int vpu_ring_buffer_length = 16;

  // TODO replace 16 and 32
  for (int x_height = 1; x_height <= 3; ++x_height) {
    for (int x_width = 1; x_width <= 3; ++x_width) {
      for (int x_channels = 32; x_channels <= 32 * 3; x_channels += 32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride) {
                    for (int output_channels = 16; output_channels <= 16 * 3; output_channels += 16) {
                      for (int input_ch_per_output = x_channels;
                           input_ch_per_output <= x_channels;
                           input_ch_per_output += 32) {
                        int output_height = CONV2D_OUTPUT_LENGTH(
                            x_height, k_height, k_v_dilation, k_v_stride);
                        int output_width = CONV2D_OUTPUT_LENGTH(
                            x_width, k_width, k_h_dilation, k_h_stride);

                        if (output_height <= 0 || output_width <= 0) continue;

                        ImageGeometry X(x_height, x_width, x_channels);
                        WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                         k_h_stride, 0, k_v_dilation,
                                         k_h_dilation);

                        std::array<int, 4> shape = {
                            {output_channels, k_height, k_width, x_channels}};
                        // flattened [output_channels][k_height][k_width][x_channels]
                        std::vector<int8_t> raw_weights(
                            (size_t)output_channels * k_height * k_width * x_channels);

                        for (size_t j = 0; j < raw_weights.size(); ++j)
                          raw_weights[j] = rng.rand<int8_t>();

                        // flattened [x_height][x_width][x_channels]
                        std::vector<int8_t> X_mem(
                            (size_t)x_height * x_width * x_channels);

                        for (size_t j = 0; j < X_mem.size(); ++j)
                          X_mem[j] = rng.rand<int8_t>();

                        int8_t pad_val =
                            rng.rand<int8_t>();  // this should be unused in
                                                 // this case

                        Conv2dReorderedWeights rw =
                            MatMulInt8::reorder_kernel_weights(
                                raw_weights.data(), shape, 8, pad_val);

                        MatMulDirectFn mmd(X, K, input_ch_per_output
                                                 //,
                                                 //  rw.weights.data(),
                                                 //  rw.weights.size()
                        );
                        mat_mul_direct_params_t p = mmd.getParams();
                        int ocg_count =
                            (output_channels + vpu_ring_buffer_length - 1) /
                            vpu_ring_buffer_length;

                        for (int ocg = 0; ocg < ocg_count; ++ocg) {
                          alignas(4) VPURingBuffer A;
                          mat_mul_direct_int8(&p, &A, X_mem.data(), ocg, rw.weights.data());

                          int chs_in_group = std::min(
                              output_channels - vpu_ring_buffer_length * ocg,
                              vpu_ring_buffer_length);

                          for (int output_chan = 0; output_chan < chs_in_group;
                               ++output_chan) {
                            int actual_output_channel =
                                output_chan + ocg * vpu_ring_buffer_length;

                            int expected_sum = 0;

                            for (int h = 0; h < k_height; ++h) {
                              for (int w = 0; w < k_width; ++w) {
                                for (int c = 0; c < input_ch_per_output; ++c) {
                                  int x = (int)X_mem[
                                      ((size_t)(k_v_dilation * h) * x_width +
                                       (k_h_dilation * w)) * x_channels + c];
                                  int t = raw_weights[
                                      (((size_t)actual_output_channel * k_height + h) *
                                           k_width + w) * x_channels + c];
                                  expected_sum += x * t;
                                }
                              }
                            }

                            int32_t v;
                            ((int16_t *)&v)[0] = A.vR[output_chan];
                            ((int16_t *)&v)[1] = A.vD[output_chan];
                            TEST_ASSERT_EQUAL(v, expected_sum);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}
/*
  Simple test to verify memory accesses.
*/
TEST(group_aggregate_fns, Test_MatMulBinaryDirectFn) {
  const int vpu_ring_buffer_length = 16;

  // TODO replace 16 and 32
  for (int x_height = 1; x_height <= 3; ++x_height) {
    for (int x_width = 1; x_width <= 3; ++x_width) {
      for (int x_channels = 256; x_channels <= 256 * 3; x_channels += 256) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride) {
                    for (int output_channels = 256; output_channels <= 256 * 3; output_channels += 256) {
                      for (int input_ch_per_output = x_channels; input_ch_per_output <= x_channels; input_ch_per_output += 256) {
                        int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                        int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);
                        if (output_height <= 0 || output_width <= 0) continue;
                        ImageGeometry X(x_height, x_width, x_channels, 1);
                        WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                         k_h_stride, 0, k_v_dilation,
                                         k_h_dilation);

                        std::array<int, 4> shape = {
                            {output_channels, k_height, k_width, x_channels}};
                        // flattened [output_channels][k_height][k_width][x_channels/8]
                        int x_bytes = x_channels / 8;
                        std::vector<int8_t> raw_weights(
                            (size_t)output_channels * k_height * k_width * x_bytes);

                        for (size_t j = 0; j < raw_weights.size(); ++j)
                          raw_weights[j] = rng.rand<int8_t>();

                        // flattened [x_height][x_width][x_channels/8]
                        std::vector<int8_t> X_mem(
                            (size_t)x_height * x_width * x_bytes);

                        for (size_t j = 0; j < X_mem.size(); ++j)
                          X_mem[j] = rng.rand<int8_t>();

                        int8_t pad_val =
                            rng.rand<int8_t>();  // this should be unused in
                                                 // this case

                        Conv2dReorderedWeights rw =
                            MatMulInt8::reorder_kernel_weights(
                                raw_weights.data(), shape, 1, pad_val);

                        MatMulBinaryDirectFn mmd(X, K, input_ch_per_output);
                        mat_mul_direct_params_t p = mmd.getParams();

                        int ocg_count =
                            (output_channels + vpu_ring_buffer_length - 1) /
                            vpu_ring_buffer_length;

                        for (int ocg = 0; ocg < ocg_count; ++ocg) {
                          alignas(4) VPURingBuffer A;
                          mat_mul_direct_binary(&p, &A, X_mem.data(), ocg, rw.weights.data());

                          int chs_in_group = std::min(
                              output_channels - vpu_ring_buffer_length * ocg,
                              vpu_ring_buffer_length);

                          for (int output_chan = 0; output_chan < chs_in_group;
                               ++output_chan) {
                            int actual_output_channel =
                                output_chan + ocg * vpu_ring_buffer_length;

                            int expected_sum = 0;

                            for (int h = 0; h < k_height; ++h) {
                              for (int w = 0; w < k_width; ++w) {
                                int input_bytes_per_output =
                                    input_ch_per_output / CHAR_BIT;
                                for (int c = 0; c < input_bytes_per_output;
                                     ++c) {
                                  int8_t x_byte = (int8_t)X_mem[
                                      ((size_t)(k_v_dilation * h) * x_width +
                                       (k_h_dilation * w)) * x_bytes + c];
                                  int8_t k_byte = (int8_t)raw_weights[
                                      (((size_t)actual_output_channel * k_height + h) *
                                           k_width + w) * x_bytes + c];
                                  accumulate_binary_bytes(&expected_sum, x_byte,
                                                          k_byte);
                                }
                              }
                            }

                            int32_t v;
                            ((int16_t *)&v)[0] = A.vR[output_chan];
                            ((int16_t *)&v)[1] = A.vD[output_chan];
                            TEST_ASSERT_EQUAL(v, expected_sum);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(group_aggregate_fns, Test_Kernel_Reordering) {
  for (int x_channels = 1; x_channels <= 6; ++x_channels) {
    for (int k_height = 1; k_height <= 6; ++k_height) {
      for (int k_width = 1; k_width <= 6; ++k_width) {
        for (int y_channels = 1; y_channels <= 6; ++y_channels) {
          std::vector<int8_t> raw_weights((size_t)y_channels * k_height * k_width * x_channels, 0);

          std::array<int, 4> shape = {
              {y_channels, k_height, k_width, x_channels}};
          int bits_per_element = 8;

          Conv2dReorderedWeights rw = MatMulInt8::reorder_kernel_weights(
              raw_weights.data(), shape, bits_per_element, 0);
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses.
*/
TEST(group_aggregate_fns, Test_Simple_MatMulDirectFn_DW) {
  const int vpu_ring_buffer_length = 16;

  std::list<std::tuple<int8_t, int8_t> > args = {
      std::tuple<int8_t, int8_t>{1, 1},  std::tuple<int8_t, int8_t>{1, 0},
      std::tuple<int8_t, int8_t>{0, 1},  std::tuple<int8_t, int8_t>{-1, 1},
      std::tuple<int8_t, int8_t>{1, -1}, std::tuple<int8_t, int8_t>{-1, -1},
  };

  for (auto arg : args) {
    int8_t kernel_fill, scratch_fill;
    std::tie(kernel_fill, scratch_fill) = arg;

    for (int x_height = 1; x_height <= 4; ++x_height) {
      for (int x_width = 1; x_width <= 4; ++x_width) {
        for (int x_channels = 4; x_channels <= 32 * 3; x_channels += 4) {
          for (int k_height = 1; k_height <= x_height; ++k_height) {
            for (int k_width = 1; k_width <= x_width; ++k_width) {
              std::array<int, 4> shape = {{1, k_height, k_width, x_channels}};
              ImageGeometry X_params(x_height, x_width, x_channels);
              WindowGeometry K_params(k_height, k_width, 1, 1, 1, 1);

              int weight_tensor_overread = 32;
              int input_tensor_overread = 32;
              std::vector<int8_t> K(k_height * k_width * x_channels + weight_tensor_overread);

              std::vector<int8_t> T(x_height * x_width * x_channels + input_tensor_overread);

              std::fill_n(K.data(), K.size(), kernel_fill);
              std::fill_n(T.data(), T.size(), scratch_fill);

              int8_t pad_val = 0;
              Conv2dReorderedWeights rw =
                  MatMulDirectFn_DW::reorder_kernel_weights(K.data(), shape,
                                                            pad_val);

              int8_t *weights = rw.weights.data();

              MatMulDirectFn_DW mmd(
                  X_params, K_params
                  // ,
                  // weights,
                  //                             sizeof(K)
              );
              mat_mul_dw_direct_params_t p = mmd.getParams();

              int ocg_count = (x_channels + vpu_ring_buffer_length - 1) /
                              vpu_ring_buffer_length;

              for (int x = 0; x < x_height - k_height + 1; ++x) {
                for (int y = 0; y < x_width - k_width + 1; ++y) {
                  for (int ocg = 0; ocg < ocg_count; ++ocg) {
                    alignas(4) VPURingBuffer A;
                    int8_t *X_mem_ch_grp = T.data() + ocg * 16;
                    mat_mul_dw_direct(&p, &A, X_mem_ch_grp, ocg, weights);

                    for (int output_chan = 0;
                         output_chan < vpu_ring_buffer_length; ++output_chan) {
                      int actual_ch = output_chan + ocg * 16;

                      if (actual_ch >= x_channels) continue;

                      int32_t v;
                      ((int16_t *)&v)[0] = A.vR[output_chan];
                      ((int16_t *)&v)[1] = A.vD[output_chan];

                      TEST_ASSERT_EQUAL(
                          k_width * k_height * (kernel_fill * scratch_fill), v);
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses.
*/
TEST(group_aggregate_fns, Test_MatMulDirectFn_DW) {
  const int vpu_ring_buffer_length = 16;

  // TODO replace 16 and 32
  for (int x_height = 1; x_height <= 4; ++x_height) {
    for (int x_width = 1; x_width <= 4; ++x_width) {
      for (int x_channels = 4; x_channels <= 32 + 4; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride) {
                    int output_height = CONV2D_OUTPUT_LENGTH(
                        x_height, k_height, k_v_dilation, k_v_stride);
                    int output_width = CONV2D_OUTPUT_LENGTH(
                        x_width, k_width, k_h_dilation, k_h_stride);

                    if (output_height <= 0 || output_width <= 0) continue;

                    ImageGeometry X(x_height, x_width, x_channels);
                    WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                     k_h_stride, 0, k_v_dilation, k_h_dilation);

                    std::array<int, 4> shape = {
                        {1, k_height, k_width, x_channels}};

                    int input_tensor_overread = 32;
                    // flattened [k_height][k_width][x_channels]
                    std::vector<int8_t> raw_weights(
                        (size_t)k_height * k_width * x_channels);

                    for (size_t j = 0; j < raw_weights.size(); ++j)
                      raw_weights[j] = rng.rand<int8_t>();

                    std::vector<int8_t> X_mem(x_height * x_width * x_channels +
                                            input_tensor_overread);

                    for (size_t j = 0; j < X_mem.size(); ++j)
                      X_mem[j] = rng.rand<int8_t>();

                    int8_t pad_val = rng.rand<int8_t>();  // this should be
                                                          // unused in this case

                    Conv2dReorderedWeights rw =
                        MatMulDirectFn_DW::reorder_kernel_weights(
                            raw_weights.data(), shape, pad_val);

                    MatMulDirectFn_DW mmd(X, K);
                    mat_mul_dw_direct_params_t p = mmd.getParams();

                    int ocg_count = (x_channels + vpu_ring_buffer_length - 1) /
                                    vpu_ring_buffer_length;

                    for (int ocg = 0; ocg < ocg_count; ++ocg) {
                      alignas(4) VPURingBuffer A;

                      // We need to dereference the pointer here so as to test
                      // the correct ocg
                      int8_t *X_mem_ch_grp = X_mem.data() + ocg * 16;
                      mat_mul_dw_direct(&p, &A, X_mem_ch_grp, ocg, rw.weights.data());

                      int chs_in_group =
                          std::min(x_channels - vpu_ring_buffer_length * ocg,
                                   vpu_ring_buffer_length);

                      for (int output_chan = 0; output_chan < chs_in_group;
                           ++output_chan) {
                        int actual_output_channel =
                            output_chan + ocg * vpu_ring_buffer_length;

                        int expected_sum = 0;

                        for (int h = 0; h < k_height; ++h) {
                          for (int w = 0; w < k_width; ++w) {
                            int x =
                                *(X_mem.data() + actual_output_channel +
                                  (k_h_dilation * w * x_channels) +
                                  (k_v_dilation * h * x_channels * x_width));

                            int t = (int)raw_weights[
                                ((size_t)h * k_width + w) * x_channels +
                                actual_output_channel];
                            expected_sum += x * t;
                          }
                        }

                        int32_t v;
                        ((int16_t *)&v)[0] = A.vR[output_chan];
                        ((int16_t *)&v)[1] = A.vD[output_chan];
                        TEST_ASSERT_EQUAL(expected_sum, v);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

TEST(group_aggregate_fns, Test_Kernel_Reordering_DW) {
  for (int x_channels = 4; x_channels <= 32; x_channels += 4) {
    for (int k_height = 1; k_height <= 6; ++k_height) {
      for (int k_width = 1; k_width <= 6; ++k_width) {
        std::vector<int8_t> raw_weights((size_t)x_channels * k_height * k_width);

        std::array<int, 4> shape = {{1, k_height, k_width, x_channels}};

        for (size_t i = 0; i < raw_weights.size(); ++i)
          raw_weights[i] = rng.rand<int8_t>();

        Conv2dReorderedWeights rw = MatMulDirectFn_DW::reorder_kernel_weights(
            raw_weights.data(), shape, 0);
      }
    }
  }
}

TEST(group_aggregate_fns, Test_MatMulDirectFn_int16) {
  const int vpu_ring_buffer_length = 16;
  int max_width = 3;

  for (int x_height = 1; x_height <= max_width; ++x_height) {
    for (int x_width = 1; x_width <= max_width; ++x_width) {
      for (int x_channels = 32; x_channels <= 32 * 3; x_channels += 32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= max_width; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= max_width; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= max_width; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= max_width; ++k_v_stride) {
                    for (int output_channels = 16; output_channels <= 16 * 3;
                         output_channels += 16) {
                      for (int input_ch_per_output = x_channels;
                           input_ch_per_output <= x_channels;
                           input_ch_per_output += 32) {
                        int output_height = CONV2D_OUTPUT_LENGTH(
                            x_height, k_height, k_v_dilation, k_v_stride);
                        int output_width = CONV2D_OUTPUT_LENGTH(
                            x_width, k_width, k_h_dilation, k_h_stride);

                        if (output_height <= 0 || output_width <= 0) continue;

                        ImageGeometry X(x_height, x_width, x_channels, 16);
                        WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                         k_h_stride, 0, k_v_dilation,
                                         k_h_dilation);

                        std::array<int, 4> shape = {
                            {output_channels, k_height, k_width, x_channels}};
                        // flattened [output_channels][k_height][k_width][x_channels]
                        std::vector<int8_t> raw_weights(
                            (size_t)output_channels * k_height * k_width * x_channels);
                        for (size_t j = 0; j < raw_weights.size(); ++j)
                            raw_weights[j] = rng.rand<int8_t>();

                        // flattened [x_height][x_width][x_channels]
                        std::vector<int16_t> X_mem(
                            (size_t)x_height * x_width * x_channels);

                        for (size_t j = 0; j < X_mem.size(); ++j)
                          X_mem[j] = rng.rand<int16_t>();

                        int8_t pad_val =
                            rng.rand<int8_t>();  // this should be unused in
                                                 // this case

                        Conv2dReorderedWeights rw =
                            MatMulInt8::reorder_kernel_weights(
                                raw_weights.data(), shape, 8, pad_val, true);

                        std::vector<int16_t> expanded_weights(raw_weights.size());

                        expand_8_to_16(expanded_weights.data(), rw.weights.data(), (int)raw_weights.size());

                        MatMulDirectFn mmd(X, K, input_ch_per_output
                                                 //,
                                                 //  rw.weights.data(),
                                                 //  rw.weights.size()
                        );
                        mat_mul_direct_params_t p = mmd.getParams();
                        int ocg_count =
                            (output_channels + vpu_ring_buffer_length - 1) /
                            vpu_ring_buffer_length;

                        for (int ocg = 0; ocg < ocg_count; ++ocg) {
                          alignas(4) VPURingBuffer A;
                          mat_mul_direct_int16(&p, &A, X_mem.data(), ocg, expanded_weights.data());

                          int chs_in_group = std::min(
                              output_channels - vpu_ring_buffer_length * ocg,
                              vpu_ring_buffer_length);

                          for (int output_chan = 0; output_chan < chs_in_group;
                               ++output_chan) {
                            int actual_output_channel =
                                output_chan + ocg * vpu_ring_buffer_length;

                            int expected_sum = 0;

                            for (int h = 0; h < k_height; ++h) {
                              for (int w = 0; w < k_width; ++w) {
                                for (int c = 0; c < input_ch_per_output; ++c) {
                                  int x = (int)X_mem[
                                      ((size_t)(k_v_dilation * h) * x_width +
                                       (k_h_dilation * w)) * x_channels + c];
                                  int t = raw_weights[
                                      (((size_t)actual_output_channel * k_height + h) *
                                           k_width + w) * x_channels + c];
                                  expected_sum += x * t;
                                }
                              }
                            }

                            int32_t v;
                            ((int16_t *)&v)[0] = A.vR[output_chan];
                            ((int16_t *)&v)[1] = A.vD[output_chan];
                            TEST_ASSERT_EQUAL(expected_sum, v);
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses.
  Disabled: crashes, see lib_nn issue tracker.
*/
TEST(group_aggregate_fns, Test_MatMulDirectFn_int16_DW) {
  // KNOWN ISSUE: fails on native (Expected -1529670 Was 466068094), not yet
  // root-caused.
  TEST_IGNORE_MESSAGE("Test_MatMulDirectFn_int16_DW fails on native");
  const int vpu_ring_buffer_length = 16;
  int max_width = 3;

  for (int x_height = 1; x_height <= 4; ++x_height) {
    for (int x_width = 1; x_width <= 4; ++x_width) {
      for (int x_channels = 4; x_channels <= 32 + 4; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= max_width; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= max_width; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= max_width; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= max_width; ++k_v_stride) {
                    int output_height = CONV2D_OUTPUT_LENGTH(
                        x_height, k_height, k_v_dilation, k_v_stride);
                    int output_width = CONV2D_OUTPUT_LENGTH(
                        x_width, k_width, k_h_dilation, k_h_stride);

                    if (output_height <= 0 || output_width <= 0) continue;

                    ImageGeometry X(x_height, x_width, x_channels, 16);
                    WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                     k_h_stride, 0, k_v_dilation, k_h_dilation);

                    std::array<int, 4> shape = {
                        {1, k_height, k_width, x_channels}};

                    int input_tensor_overread = 32;
                    // flattened [k_height][k_width][x_channels]
                    std::vector<int8_t> raw_weights(
                        (size_t)k_height * k_width * x_channels);

                    for (size_t j = 0; j < raw_weights.size(); ++j)
                        raw_weights[j] = rng.rand<int8_t>();

                    std::vector<int16_t> X_mem(x_height * x_width * x_channels +
                                            input_tensor_overread);

                    for (size_t j = 0; j < X_mem.size(); ++j)
                      X_mem[j] = rng.rand<int16_t>();

                    int16_t pad_val = rng.rand<int16_t>();  // this should be
                                                          // unused in this case

                    Conv2dReorderedWeights rw =
                        MatMulDirectFn_DW::reorder_kernel_weights(
                            raw_weights.data(), shape, pad_val);

                    std::vector<int16_t> expanded_weights(raw_weights.size());

                    expand_8_to_16(expanded_weights.data(), raw_weights.data(), (int)raw_weights.size());

                    MatMulDirectFn_DW mmd(X, K);
                    mat_mul_dw_direct_params_t p = mmd.getParams();

                    int ocg_count = (x_channels + vpu_ring_buffer_length - 1) /
                                    vpu_ring_buffer_length;

                    for (int ocg = 0; ocg < ocg_count; ++ocg) {
                      alignas(4) VPURingBuffer A;

                      // We need to dereference the pointer here so as to test
                      // the correct ocg
                      int16_t *X_mem_ch_grp = X_mem.data() + ocg * 16;
                      mat_mul_dw_direct_int16(&p, &A, X_mem_ch_grp, ocg, expanded_weights.data());

                      int chs_in_group =
                          std::min(x_channels - vpu_ring_buffer_length * ocg,
                                   vpu_ring_buffer_length);

                      for (int output_chan = 0; output_chan < chs_in_group;
                           ++output_chan) {
                        int actual_output_channel =
                            output_chan + ocg * vpu_ring_buffer_length;

                        int expected_sum = 0;

                        for (int h = 0; h < k_height; ++h) {
                          for (int w = 0; w < k_width; ++w) {
                            int x =
                                *(X_mem.data() + actual_output_channel +
                                  (k_h_dilation * w * x_channels) +
                                  (k_v_dilation * h * x_channels * x_width));

                            int t = (int)raw_weights[
                                ((size_t)h * k_width + w) * x_channels +
                                actual_output_channel];
                            expected_sum += x * t;
                          }
                        }

                        int32_t v;
                        ((int16_t *)&v)[0] = A.vR[output_chan];
                        ((int16_t *)&v)[1] = A.vD[output_chan];
                        TEST_ASSERT_EQUAL(expected_sum, v);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // extern "C"
