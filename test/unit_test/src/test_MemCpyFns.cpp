// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <cstring>

#include "MemCpyFn.hpp"
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

TEST_GROUP(group_mem_cpy_fns);
TEST_SETUP(group_mem_cpy_fns) {}
TEST_TEAR_DOWN(group_mem_cpy_fns) {}
TEST_GROUP_RUNNER(group_mem_cpy_fns) {
  RUN_TEST_CASE(group_mem_cpy_fns, Test_ImToColValid);
  RUN_TEST_CASE(group_mem_cpy_fns, Test_ImToColPadded);
  RUN_TEST_CASE(group_mem_cpy_fns, Test_DerefInputFn);
}

// TODO binary tests for ImToColValid
TEST(group_mem_cpy_fns, Test_ImToColValid) {
  for (int x_height = 1; x_height <= 8; ++x_height) {
    for (int x_width = 1; x_width <= 8; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            // the number of channels the kernel is going to be copying
            // TODO put input_ch_per_output in K.channels i.e. the output
            // channels fo K
            for (int input_ch_per_output = 4; input_ch_per_output <= x_channels;
                 input_ch_per_output += 4) {
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

                      unsigned k_depth = 1;  //(input_ch_per_output);

                      WindowGeometry K(k_height, k_width, k_depth, 0, 0,
                                       k_v_stride, k_h_stride, 1, k_v_dilation,
                                       k_h_dilation);

                      ImToColValid cpy(X, K, input_ch_per_output);
                      memcpyfn_imtocol_valid_params_t p = cpy.getParams();
                      size_t scratch_bytes = cpy.get_scratch_bytes();
                      int overread_bytes = cpy.get_overread_bytes();

                      std::vector<int8_t> T(scratch_bytes);
                      std::vector<int8_t> X_mem(x_width * x_height * x_channels + overread_bytes);

                      // The count of x channels from whick a memcopy could
                      // start
                      int output_ch_starts = x_channels / input_ch_per_output;

                      for (int output_h = 0; output_h < output_height;
                           ++output_h) {
                        for (int output_w = 0; output_w < output_width;
                             ++output_w) {
                          for (int output_c = 0; output_c < output_ch_starts;
                               output_c +=
                               4) {  // only test from aligned memory addresses

                            for (int j = 0; j < sizeof X_mem; ++j)
                              X_mem[j] = rng.rand<int8_t>();

                            std::memset(T.data(), 0x55, sizeof T);

                            memcpyfn_imtocol_valid(&p, T.data(), X_mem.data(), output_h, output_w, output_c);

                            int t_idx = 0;

                            for (int kh = 0; kh < k_height; ++kh) {
                              for (int kw = 0; kw < k_width; ++kw) {
                                for (int kc = 0; kc < input_ch_per_output;
                                     ++kc) {
                                  int t = (int)T[t_idx++];
                                  // TODO use Aarons code here
                                  int x = (int)X_mem[(kh * k_v_dilation +
                                                      k_v_stride * output_h) *
                                                         x_channels * x_width +
                                                     (kw * k_h_dilation +
                                                      k_h_stride * output_w) *
                                                         x_channels +
                                                     kc + output_c];

                                  TEST_ASSERT_EQUAL(t, x);
                                }
                              }
                            }
                            for (; t_idx < scratch_bytes; ++t_idx) {
                              TEST_ASSERT_EQUAL(0, T[t_idx]);
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
}

TEST(group_mem_cpy_fns, Test_ImToColPadded) {
  // TODO use above generator class

  for (int x_height = 1; x_height <= 4; ++x_height) {
    for (int x_width = 1; x_width <= 4; ++x_width) {
      for (int x_channels = 1; x_channels <= 8; x_channels += 1) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                    for (int input_ch_per_output = 1;
                         input_ch_per_output <= x_channels;
                         input_ch_per_output += 1) {
                      for (int top_p = 0; top_p <= 1; ++top_p) {
                        for (int bot_p = 0; bot_p <= 1; ++bot_p) {
                          for (int left_p = 0; left_p <= 1; ++left_p) {
                            for (int right_p = 0; right_p <= 1; ++right_p) {
                              // TODO can we make padding_t int types?
                              padding_t padding = {
                                  (int16_t)top_p, (int16_t)bot_p,
                                  (int16_t)left_p, (int16_t)right_p};

                              int padded_x_height =
                                  padding.top + padding.bottom + x_height;
                              int padded_x_width =
                                  padding.left + padding.right + x_width;

                              int output_height = CONV2D_OUTPUT_LENGTH(
                                  padded_x_height, k_height, k_v_dilation,
                                  k_v_stride);
                              int output_width = CONV2D_OUTPUT_LENGTH(
                                  padded_x_width, k_width, k_h_dilation,
                                  k_h_stride);

                              if (output_height <= 0 || output_width <= 0)
                                continue;

                              ImageGeometry X(x_height, x_width, x_channels);

                              WindowGeometry K(k_height, k_width, 0, 0, 0,
                                               k_v_stride, k_h_stride, 0,
                                               k_v_dilation, k_h_dilation);

                              int8_t pad_val = 0x55;

                              ImToColPadded cpy(
                                  X, K, padding, input_ch_per_output, pad_val);
                              memcpyfn_imtocol_padded_params_t p = cpy.getParams();
                              size_t scratch_bytes =
                                  cpy.get_scratch_bytes();  // TODO add test
                                                            // that crashes when
                                                            // this is one less
                              int overread_bytes =
                                  cpy.get_overread_bytes();  // TODO add test
                                                             // that crashes
                                                             // when this is one
                                                             // less

                              std::vector<int8_t> T(scratch_bytes);
                              std::vector<int8_t> X_mem(x_height * x_width *
                                                        x_channels);

                              // create an explicitly padded version of X
                              std::vector<int8_t> X_mem_padded(
                                  padded_x_height * padded_x_width *
                                  x_channels);

                              for (size_t i = 0; i < X_mem.size(); ++i)
                                X_mem[i] = rng.rand<int8_t>();

                              std::memset(X_mem_padded.data(), pad_val,
                                          X_mem_padded.size());

                              for (int h = 0; h < x_height; h++) {
                                for (int w = 0; w < x_width; w++) {
                                  for (int c = 0; c < x_channels; c++) {
                                    X_mem_padded[((h + padding.top) *
                                                      padded_x_width +
                                                  (w + padding.left)) *
                                                     x_channels +
                                                 c] =
                                        X_mem[(h * x_width + w) * x_channels +
                                              c];
                                  }
                                }
                              }

                              std::vector<int8_t> X_mem_with_overread(
                                  X_mem.size() + overread_bytes);

                              std::memcpy(X_mem_with_overread.data(),
                                          X_mem.data(), X_mem.size());
                              for (size_t j = X_mem.size();
                                   j < X_mem_with_overread.size(); ++j)
                                X_mem_with_overread[j] = rng.rand<int8_t>();

                              int output_ch_starts =
                                  x_channels / input_ch_per_output;

                              for (int output_h = 0; output_h < output_height;
                                   ++output_h) {
                                for (int output_w = 0; output_w < output_width;
                                     ++output_w) {
                                  for (int output_c = 0;
                                       output_c < output_ch_starts;
                                       ++output_c) {  // only test from aligned
                                                      // memory addresses

                                    std::memset(T.data(), 0xaa, T.size());

                                    memcpyfn_imtocol_padded(&p, T.data(),
                                                   X_mem_with_overread.data(),
                                                   output_h, output_w,
                                                   output_c);

                                    int t_idx = 0;

                                    for (int kh = 0; kh < k_height; ++kh) {
                                      for (int kw = 0; kw < k_width; ++kw) {
                                        for (int kc = 0;
                                             kc < input_ch_per_output; ++kc) {
                                          int t = (int)T[t_idx++];

                                          // TODO use Aarons code here
                                          int x = (int)X_mem_padded
                                              [((kh * k_v_dilation +
                                                 k_v_stride * output_h) *
                                                    padded_x_width +
                                                (kw * k_h_dilation +
                                                 k_h_stride * output_w)) *
                                                   x_channels +
                                               kc + output_c];
                                          TEST_ASSERT_EQUAL(t, x);
                                        }
                                      }
                                    }
                                    for (; t_idx < scratch_bytes; ++t_idx) {
                                      TEST_ASSERT_EQUAL(0, T[t_idx]);
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
        }
      }
    }
  }
}

TEST(group_mem_cpy_fns, Test_DerefInputFn) {
  int k_h_dilation = 1;
  int k_v_dilation = 1;

  for (int x_height = 1; x_height <= 10; ++x_height) {
    for (int x_width = 1; x_width <= 10; ++x_width) {
      for (int x_channels = 1; x_channels <= 8; x_channels += 1) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_stride = 1; k_h_stride <= 4; ++k_h_stride) {
              for (int k_v_stride = 1; k_v_stride <= 4; ++k_v_stride) {
                int output_height = CONV2D_OUTPUT_LENGTH(
                    x_height, k_height, k_v_dilation, k_v_stride);
                int output_width = CONV2D_OUTPUT_LENGTH(
                    x_width, k_width, k_h_dilation, k_h_stride);

                if (output_height <= 0 || output_width <= 0) continue;

                ImageGeometry X(x_height, x_width, x_channels);
                WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                 k_h_stride, 0, k_v_dilation, k_h_dilation);

                DerefInputFn memcpy(X,K);
                memcpyfn_deref_params_t m = memcpy.getParams();

                std::vector<int8_t> X_mem(x_height * x_width * x_channels);

                for (size_t j = 0; j < X_mem.size(); ++j)
                  X_mem[j] = rng.rand<int8_t>();

                for (int h = 0; h < output_height; ++h) {
                  for (int w = 0; w < output_width; ++w) {
                    for (int c = 0; c < x_channels; ++c) {
                      int8_t *p = memcpyfn_deref(&m, 0, X_mem.data(), h, w, c);
                      int x = (int)X_mem[(k_v_stride * h * x_width +
                                          k_h_stride * w) *
                                             x_channels +
                                         c];
                      TEST_ASSERT_EQUAL((int)*p, x);
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
