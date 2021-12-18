#include <list>
#include <tuple>

#include "AggregateFn.hpp"
#include "Rand.hpp"

extern "C" {
#include "tst_common.h"
#include "unity.h"
}
using namespace nn;
using namespace nn::test;

static auto rng = test::Rand(42);

/*
  Simple test to verify memory accesses
*/
void Test_SimpleMatMulInt8() {
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

        MatMulInt8::Params p(output_channel_count, input_bytes);
        MatMulInt8 mm(&p);
        mm.setWeights((int8_t *)K);

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
void Test_SimpleMatMulBinary() {
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

        alignas(4) int8_t K[kernel_bytes];
        alignas(4) int8_t T[scratch_bytes];

        MatMulBase::Params p(output_channel_count, input_bytes);
        MatMulBinary mm(&p);
        mm.setWeights((int8_t *)K);

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
void Test_MatMulInt8() {
  const int vpu_bytes = 32;
  const int vpu_ring_buffer_length = 16;

  for (int input_bytes = 4; input_bytes < 128; input_bytes += 4) {
    for (int output_channel_count = 1; output_channel_count < 48;
         ++output_channel_count) {
      int k_height = 1;
      int k_width = 1;  // to make things easy

      std::array<int, 4> shape = {
          {output_channel_count, k_height, k_width, input_bytes}};
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

      MatMulInt8::Params p(output_channel_count,
                           input_bytes);  // reordered_weights
      MatMulInt8 mm(&p);
      mm.setWeights((int8_t *)reordered_weights);
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
void Test_MatMulBinary() {
  const int vpu_bytes = 32;
  const int vpu_ring_buffer_length = 16;

  for (int input_bytes = 4; input_bytes < 128; input_bytes += 4) {
    for (int output_channel_count = 8; output_channel_count < 48;
         output_channel_count += 8) {
      int k_height = 1;
      int k_width = 1;  // to make things easy

      std::array<int, 4> shape = {
          {output_channel_count, k_height, k_width, input_bytes}};
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

      int expected[output_channel_count];
      for (int i = 0; i < output_channel_count; i++) {
        expected[i] = 0;
        for (int j = 0; j < sizeof T; ++j) {
          accumulate_binary_bytes(&(expected[i]), ((int8_t *)raw_weights)[j],
                                  T[j]);
        }
      }

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

      MatMulInt8::Params p(output_channel_count,
                           input_bytes);  // reordered_weights
      MatMulInt8 mm(&p);
      mm.setWeights((int8_t *)reordered_weights);
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
void Test_Simple_MatMulDirectFn() {
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

                MatMulDirectFn::Params p(
                    X_params, K_params,
                    x_channels);  // weights, (int)(y_channels * k_height *
                                  // k_width * x_channels)
                MatMulDirectFn mmd(&p);
                mmd.setWeights(weights);

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
void Test_Simple_MatMulBinaryDirectFn() {
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

                alignas(4)
                    int8_t K[y_channels][k_height][k_width][x_channels / 8];
                alignas(4) int8_t T[x_height][x_width][x_channels / 8];

                int8_t *weights =
                    (int8_t *)K;  // todo we will switch to usnig the boggler

                MatMulBinaryDirectFn::Params p(X_params, K_params, x_channels);
                MatMulBinaryDirectFn mmd(&p);
                mmd.setWeights(weights);

                std::fill_n((int8_t *)K, sizeof K, kernel_fill);
                std::fill_n((int8_t *)T, x_height * x_width * x_channels / 8,
                            scratch_fill);

                int expected = 0;
                accumulate_binary_bytes(&(expected), kernel_fill, scratch_fill);
                expected *= (k_height * k_width * x_channels / CHAR_BIT);

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
void Test_MatMulDirectFn() {
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

                        ImageGeometry X(x_height, x_width, x_channels);
                        WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                         k_h_stride, 0, k_v_dilation,
                                         k_h_dilation);

                        std::array<int, 4> shape = {
                            {output_channels, k_height, k_width, x_channels}};
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

                        MatMulDirectFn::Params p(X, K, input_ch_per_output
                                                 //,
                                                 //  rw.weights.data(),
                                                 //  rw.weights.size()
                        );
                        MatMulDirectFn mmd(&p);
                        mmd.setWeights(rw.weights.data());
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
void Test_MatMulBinaryDirectFn() {
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
                    for (int output_channels = 256; output_channels <= 256 * 3;
                         output_channels += 256) {
                      for (int input_ch_per_output = x_channels;
                           input_ch_per_output <= x_channels;
                           input_ch_per_output += 256) {
                        int output_height = CONV2D_OUTPUT_LENGTH(
                            x_height, k_height, k_v_dilation, k_v_stride);
                        int output_width = CONV2D_OUTPUT_LENGTH(
                            x_width, k_width, k_h_dilation, k_h_stride);

                        if (output_height <= 0 || output_width <= 0) continue;

                        ImageGeometry X(x_height, x_width, x_channels, 1);
                        WindowGeometry K(k_height, k_width, 0, 0, 0, k_v_stride,
                                         k_h_stride, 0, k_v_dilation,
                                         k_h_dilation);

                        std::array<int, 4> shape = {
                            {output_channels, k_height, k_width, x_channels}};
                        alignas(4) int8_t raw_weights[output_channels][k_height]
                                                     [k_width][x_channels / 8];

                        for (int j = 0; j < sizeof raw_weights; ++j)
                          ((int8_t *)raw_weights)[j] = rng.rand<int8_t>();

                        alignas(4)
                            int8_t X_mem[x_height][x_width][x_channels / 8];

                        for (int j = 0; j < sizeof X_mem; ++j)
                          ((int8_t *)X_mem)[j] = rng.rand<int8_t>();

                        int8_t pad_val =
                            rng.rand<int8_t>();  // this should be unused in
                                                 // this case

                        Conv2dReorderedWeights rw =
                            MatMulInt8::reorder_kernel_weights(
                                (int8_t *)raw_weights, shape, 1, pad_val);

                        MatMulDirectFn::Params p(X, K, input_ch_per_output);
                        MatMulBinaryDirectFn mmd(&p);

                        mmd.setWeights(rw.weights.data());
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
                                int input_bytes_per_output =
                                    input_ch_per_output / CHAR_BIT;
                                for (int c = 0; c < input_bytes_per_output;
                                     ++c) {
                                  int8_t x_byte =
                                      (int8_t)X_mem[k_v_dilation * h]
                                                   [k_h_dilation * w][c];
                                  int8_t k_byte =
                                      (int8_t)raw_weights[actual_output_channel]
                                                         [h][w][c];
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

void Test_Kernel_Reordering() {
  for (int x_channels = 1; x_channels <= 6; ++x_channels) {
    for (int k_height = 1; k_height <= 6; ++k_height) {
      for (int k_width = 1; k_width <= 6; ++k_width) {
        for (int y_channels = 1; y_channels <= 6; ++y_channels) {
          int8_t raw_weights[y_channels][k_height][k_width][x_channels];

          std::array<int, 4> shape = {
              {y_channels, k_height, k_width, x_channels}};
          int bits_per_element = 8;

          memset(raw_weights, 0, sizeof raw_weights);

          Conv2dReorderedWeights rw = MatMulInt8::reorder_kernel_weights(
              (int8_t *)raw_weights, shape, bits_per_element, 0);
        }
      }
    }
  }
}

/*
  Simple test to verify memory accesses.
*/
void Test_Simple_MatMulDirectFn_DW() {
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
              alignas(4) int8_t
                  K[k_height * k_width * x_channels + weight_tensor_overread];

              alignas(4) int8_t
                  T[x_height * x_width * x_channels + input_tensor_overread];

              std::fill_n((int8_t *)K, sizeof K, kernel_fill);
              std::fill_n((int8_t *)T, sizeof T, scratch_fill);

              int8_t pad_val = 0;
              Conv2dReorderedWeights rw =
                  MatMulDirectFn_DW::reorder_kernel_weights((int8_t *)K, shape,
                                                            pad_val);

              int8_t *weights = rw.weights.data();

              MatMulDirectFn_DW::Params p(
                  X_params, K_params
                  // ,
                  // weights,
                  //                             sizeof(K)
              );
              MatMulDirectFn_DW mmd(&p);
              mmd.setWeights(weights);

              int ocg_count = (x_channels + vpu_ring_buffer_length - 1) /
                              vpu_ring_buffer_length;

              for (int x = 0; x < x_height - k_height + 1; ++x) {
                for (int y = 0; y < x_width - k_width + 1; ++y) {
                  for (int ocg = 0; ocg < ocg_count; ++ocg) {
                    alignas(4) VPURingBuffer A;
                    int8_t *X_mem_ch_grp = T + ocg * 16;
                    mmd.aggregate_fn(&A, X_mem_ch_grp, ocg);

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
void Test_MatMulDirectFn_DW() {
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
                    alignas(4)
                        int8_t raw_weights[k_height][k_width][x_channels];

                    for (int j = 0; j < sizeof raw_weights; ++j)
                      ((int8_t *)raw_weights)[j] = rng.rand<int8_t>();

                    alignas(4) int8_t X_mem[x_height * x_width * x_channels +
                                            input_tensor_overread];

                    for (int j = 0; j < sizeof X_mem; ++j)
                      ((int8_t *)X_mem)[j] = rng.rand<int8_t>();

                    int8_t pad_val = rng.rand<int8_t>();  // this should be
                                                          // unused in this case

                    Conv2dReorderedWeights rw =
                        MatMulDirectFn_DW::reorder_kernel_weights(
                            (int8_t *)raw_weights, shape, pad_val);

                    MatMulDirectFn_DW::Params p(
                        X, K
                        // , rw.weights.data(),
                        //                             rw.weights.size()
                    );
                    MatMulDirectFn_DW mmd(&p);
                    mmd.setWeights(rw.weights.data());

                    int ocg_count = (x_channels + vpu_ring_buffer_length - 1) /
                                    vpu_ring_buffer_length;

                    for (int ocg = 0; ocg < ocg_count; ++ocg) {
                      alignas(4) VPURingBuffer A;

                      // We need to dereference the pointer here so as to test
                      // the correct ocg
                      int8_t *X_mem_ch_grp = X_mem + ocg * 16;
                      mmd.aggregate_fn(&A, (int8_t *)X_mem_ch_grp, ocg);

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
                                *(X_mem + actual_output_channel +
                                  (k_h_dilation * w * x_channels) +
                                  (k_v_dilation * h * x_channels * x_width));

                            int t =
                                (int)raw_weights[h][w][actual_output_channel];
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

void Test_Kernel_Reordering_DW() {
  for (int x_channels = 4; x_channels <= 32; x_channels += 4) {
    for (int k_height = 1; k_height <= 6; ++k_height) {
      for (int k_width = 1; k_width <= 6; ++k_width) {
        int8_t raw_weights[x_channels][k_height][k_width][1];

        std::array<int, 4> shape = {{1, k_height, k_width, x_channels}};

        memset(raw_weights, 0, sizeof raw_weights);

        for (int i = 0; i < sizeof raw_weights; ++i)
          ((int8_t *)raw_weights)[i] = rng.rand<int8_t>();

        Conv2dReorderedWeights rw = MatMulDirectFn_DW::reorder_kernel_weights(
            (int8_t *)raw_weights, shape, 0);
      }
    }
  }
}

extern "C" void test_aggregate_fns();
void test_aggregate_fns() {
  UNITY_SET_FILE();
  RUN_TEST(Test_SimpleMatMulInt8);
  RUN_TEST(Test_SimpleMatMulBinary);
  RUN_TEST(Test_MatMulInt8);
  RUN_TEST(Test_MatMulBinary);
  RUN_TEST(Test_Simple_MatMulDirectFn);
  RUN_TEST(Test_Simple_MatMulBinaryDirectFn);
  RUN_TEST(Test_MatMulDirectFn);
  RUN_TEST(Test_MatMulBinaryDirectFn);
  RUN_TEST(Test_Kernel_Reordering);
  RUN_TEST(Test_Simple_MatMulDirectFn_DW);
  RUN_TEST(Test_MatMulDirectFn_DW);
  RUN_TEST(Test_Kernel_Reordering_DW);
}
