#include <list>
#include <tuple>

#include "AggregateFn.hpp"
#include "Rand.hpp"
#include "gtest/gtest.h"

namespace nn {

static auto rng = test::Rand(42);

class Test_SimpleMatMulInt8 : public ::testing::Test {};
/*
  Simple test to verify memory accesses
*/
TEST_F(Test_SimpleMatMulInt8, BasicTest) {
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

        alignas(4) int8_t K[kernel_bytes];
        alignas(4) int8_t T[scratch_bytes];

        MatMulInt8::Params p(output_channel_count, input_bytes, (int8_t *)K);
        MatMulInt8 mm(&p);

        std::fill_n(K, kernel_bytes, kernel_fill);
        std::fill_n(T, scratch_bytes, scratch_fill);

        int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) /
                        vpu_ring_buffer_length;

        for (int ocg = 0; ocg < ocg_count; ++ocg) {
          alignas(4) VPURingBuffer A;
          mm.aggregate_fn(&A, T, ocg);

          int c;
          if ((ocg + 1) * vpu_ring_buffer_length < output_channel_count)
            c = vpu_ring_buffer_length;
          else
            c = output_channel_count % vpu_ring_buffer_length;

          for (int output_chan = 0; output_chan < c; ++output_chan) {
            int32_t v;
            ((int16_t *)&v)[0] = A.vR[output_chan];
            ((int16_t *)&v)[1] = A.vD[output_chan];

            EXPECT_EQ(scratch_bytes * (kernel_fill * scratch_fill), v);
          }
        }
      }
    }
  }
}

class Test_MatMulInt8 : public ::testing::Test {};
/*
  Simple test to verify memory accesses
*/
TEST_F(Test_MatMulInt8, BasicTest) {
  const int vpu_bytes = 32;
  const int vpu_ring_buffer_length = 16;

  for (int input_bytes = 4; input_bytes < 128; input_bytes += 4) {
    for (int output_channel_count = 1; output_channel_count < 48;
         ++output_channel_count) {
      int k_height = 1;
      int k_width = 1;  // to make things easy

      std::array<int, 4> shape = {output_channel_count, k_height, k_width,
                                  input_bytes};
      alignas(4) int8_t raw_weights[output_channel_count][k_height][k_width]
                                   [input_bytes];
      assert(sizeof raw_weights == input_bytes * output_channel_count);

      for (auto j = 0; j < sizeof raw_weights; ++j)
        ((int8_t *)raw_weights)[j] = rng.rand<int8_t>();

      int scratch_bytes = MatMulInt8::get_scratch_mem_bytes(input_bytes);

      int8_t pad_val = rng.rand<int8_t>();

      Conv2dReorderedWeights rw = MatMulInt8::reorder_kernel_weights(
          (int8_t *)raw_weights, shape, 8, pad_val);

      alignas(4) int8_t T[scratch_bytes];

      for (int j = 0; j < sizeof T; ++j) T[j] = rng.rand<int8_t>();

      int accu_modifier[output_channel_count];  //=0

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

      alignas(4) int8_t reordered_weights[rw.weights.size()];
      std::memcpy(reordered_weights, rw.weights.data(), rw.weights.size());

      MatMulInt8::Params p(output_channel_count, input_bytes,
                           reordered_weights);
      MatMulInt8 mm(&p);
      int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) /
                      vpu_ring_buffer_length;

      for (int ocg = 0; ocg < ocg_count; ++ocg) {
        alignas(4) VPURingBuffer A;
        mm.aggregate_fn(&A, T, ocg);

        int chs_in_group =
            std::min(output_channel_count - output_channel_count * ocg,
                     vpu_ring_buffer_length);

        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan) {
          int actual_output_channel =
              output_chan + ocg * vpu_ring_buffer_length;

          int expected_sum = 0;
          for (int b = 0; b < input_bytes; b++)
            expected_sum +=
                ((int)raw_weights[actual_output_channel][0][0][b] * (int)T[b]);

          int32_t v;
          ((int16_t *)&v)[0] = A.vR[output_chan];
          ((int16_t *)&v)[1] = A.vD[output_chan];

          EXPECT_EQ(v - accu_modifier[actual_output_channel], expected_sum);
        }
      }
    }
  }
}

class Test_Simple_MatMulDirectFn : public ::testing::Test {};
/*
  Simple test to verify memory accesses.
*/
TEST_F(Test_Simple_MatMulDirectFn, BasicTest) {
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

                alignas(4) int8_t K[y_channels][k_height][k_width][x_channels];
                alignas(4) int8_t T[x_height][x_width][x_channels];

                int8_t *weights =
                    (int8_t *)K;  // todo we will switch to usnig the boggler

                MatMulDirectFn::Params p(X_params, K_params, x_channels,
                                         weights);
                MatMulDirectFn mmd(&p);

                std::fill_n((int8_t *)K, sizeof K, kernel_fill);
                std::fill_n((int8_t *)T, x_height * x_width * x_channels,
                            scratch_fill);

                int ocg_count = (y_channels + vpu_ring_buffer_length - 1) /
                                vpu_ring_buffer_length;

                for (int x = 0; x < x_height - k_height + 1; ++x) {
                  for (int y = 0; y < x_width - k_width + 1; ++y) {
                    for (int ocg = 0; ocg < ocg_count; ++ocg) {
                      alignas(4) VPURingBuffer A;
                      mmd.aggregate_fn(&A, (int8_t *)T, ocg);

                      for (int output_chan = 0;
                           output_chan < vpu_ring_buffer_length;
                           ++output_chan) {
                        int32_t v;
                        ((int16_t *)&v)[0] = A.vR[output_chan];
                        ((int16_t *)&v)[1] = A.vD[output_chan];

                        EXPECT_EQ(k_width * k_height * x_channels *
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

class Test_MatMulDirectFn : public ::testing::Test {};
/*
  Simple test to verify memory accesses.
*/
TEST_F(Test_MatMulDirectFn, BasicTest) {
  const int vpu_ring_buffer_length = 16;

  // TODO replace 16 and 32
  for (int x_height = 1; x_height <= 5; ++x_height) {
    for (int x_width = 1; x_width <= 5; ++x_width) {
      for (int x_channels = 32; x_channels <= 32 * 3; x_channels += 32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride) {
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

                        // std::cout << "x_height: " << x_height
                        //           << " x_width: " << x_width
                        //           << " x_channels: " << x_channels
                        //           << " k_height: " << k_height
                        //           << " k_width: " << k_width
                        //           << " k_h_dilation: " << k_h_dilation
                        //           << " k_v_dilation: " << k_v_dilation
                        //           << " output_channels: " << output_channels
                        //           << " input_ch_per_output: " <<
                        //           input_ch_per_output
                        //           << std::endl;
                        ImageGeometry X(x_height, x_width, x_channels);
                        WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                         k_h_stride, 0, k_v_dilation,
                                         k_h_dilation);

                        std::array<int, 4> shape = {output_channels, k_height,
                                                    k_width, x_channels};
                        alignas(4) int8_t raw_weights[output_channels][k_height]
                                                     [k_width][x_channels];

                        for (int j = 0; j < sizeof raw_weights; ++j)
                          ((int8_t *)raw_weights)[j] = rng.rand<int8_t>();

                        alignas(4) int8_t X_mem[x_height][x_width][x_channels];

                        for (int j = 0; j < sizeof X_mem; ++j)
                          ((int8_t *)X_mem)[j] = rng.rand<int8_t>();

                        int8_t pad_val =
                            rng.rand<int8_t>();  // this should be unused in
                                                 // this case

                        Conv2dReorderedWeights rw =
                            MatMulInt8::reorder_kernel_weights(
                                (int8_t *)raw_weights, shape, 8, pad_val);

                        MatMulDirectFn::Params p(X, K, input_ch_per_output,
                                                 rw.weights.data());
                        MatMulDirectFn mmd(&p);

                        int ocg_count =
                            (output_channels + vpu_ring_buffer_length - 1) /
                            vpu_ring_buffer_length;

                        for (int ocg = 0; ocg < ocg_count; ++ocg) {
                          alignas(4) VPURingBuffer A;
                          mmd.aggregate_fn(&A, (int8_t *)X_mem, ocg);

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
                                  int x = (int)X_mem[k_v_dilation * h]
                                                    [k_h_dilation * w][c];
                                  int t = raw_weights[actual_output_channel][h]
                                                     [w][c];
                                  expected_sum += x * t;
                                }
                              }
                            }

                            int32_t v;
                            ((int16_t *)&v)[0] = A.vR[output_chan];
                            ((int16_t *)&v)[1] = A.vD[output_chan];
                            EXPECT_EQ(v, expected_sum);
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

class Test_Kernel_Reordering : public ::testing::Test {};

TEST_F(Test_Kernel_Reordering, BasicTest) {
  for (int x_channels = 1; x_channels <= 6; ++x_channels) {
    for (int k_height = 1; k_height <= 6; ++k_height) {
      for (int k_width = 1; k_width <= 6; ++k_width) {
        for (int y_channels = 1; y_channels <= 6; ++y_channels) {
          int8_t raw_weights[y_channels][k_height][k_width][x_channels];

          std::array<int, 4> shape = {y_channels, k_height, k_width,
                                      x_channels};
          int bits_per_element = 8;

          memset(raw_weights, 0, sizeof raw_weights);

          Conv2dReorderedWeights rw = MatMulInt8::reorder_kernel_weights(
              (int8_t *)raw_weights, shape, bits_per_element, 0);
        }
      }
    }
  }
}

}  // namespace nn
