#include "gtest/gtest.h"
#include <cstring>
#include <array>
#include <iostream>
#include <algorithm>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include <tuple>
#include <list>

namespace {

  int pseudo_rand(int *seed){
    const int a = 1013904223;
    const int c = 1664525;
    *seed = (int)((long long)a * *seed + c);
    return *seed;
  }

  class Test_Im_to_col_valid: public ::testing::Test {};

  TEST_F(Test_Im_to_col_valid, BasicTest) {

    {
      int x_width = 12;
      int x_height = 12;
      int x_channels = 3;
      int k_width = 3;
      int k_height = 3;
      int k_channels = 8;

      ImageParams X(x_width, x_height, x_channels, 8); 
      WindowGeometry K (k_width, k_height, 1, 1, 1, 1);
      ImageParams Y(X, K, k_channels);

      Im_to_col_valid cpy(X, Y, K);
      size_t scratch_bytes = cpy.get_scratch_bytes();

      int overread_bytes = 0; cpy.get_overread_bytes();

      int8_t T[scratch_bytes];
      int8_t X_mem[x_width * x_height * x_channels + overread_bytes];

      int seed = 42;

      for(auto h = 0; h < x_height - k_height; ++h){
        for(auto w = 0 ; w < x_width - k_width; ++w){

          for(auto j = 0; j < x_width * x_height * x_channels; ++j)
            X_mem[j] = (int8_t)pseudo_rand(&seed);
          
          cpy.memcopy_fn(T, X_mem, h, w);

          int j = 0;
          for(int kh = 0; kh < k_height; ++kh){
            for(int kw = 0 ; kw < k_width; ++kw){
              for(int c = 0 ; c < x_channels; ++c){
                EXPECT_EQ(T[j++], X_mem[(kh+h)*x_channels*x_height + (kw+w)*x_channels + c]);
              }
            }
          }
          for(; j < scratch_bytes; ++j){
            EXPECT_EQ(0, T[j++]);
          }
        }
      }
    }
  }

  class Test_Im_to_col_complete: public ::testing::Test {};

  TEST_F(Test_Im_to_col_complete, BasicTest) {

    {
      int x_width = 12;
      int x_height = 12;
      int x_channels = 3;
      int k_width = 3;
      int k_height = 3;
      int k_channels = 8;

      ImageParams X(x_width, x_height, x_channels, 8); 
      WindowGeometry K (k_width, k_height, 1, 1, 1, 1);
      ImageParams Y(X, K, k_channels);

      Im_to_col_valid cpy(X, Y, K);
      size_t scratch_bytes = cpy.get_scratch_bytes();

      int overread_bytes = cpy.get_overread_bytes();

      int8_t T[scratch_bytes];
      int8_t X_mem[x_width * x_height * x_channels + overread_bytes];

      for(auto h = 0; h < x_height - k_height; ++h){
        for(auto w = 0 ; w < x_width - k_width; ++w){

          std::memset(X_mem, 0, sizeof X_mem);
          std::memset(T, 0x55, sizeof T);
          
          cpy.memcopy_fn(T, X_mem, h, w);
          for(auto j = 0; j < scratch_bytes; ++j)
            EXPECT_EQ(0, T[j++]);
        }
      }
    }
  }


  class Test_MatMulFn: public ::testing::Test {};
  /*
    Simple test to verify memory accesses
  */
  TEST_F(Test_MatMulFn, BasicTest) {
    const int vpu_bytes = 32;
    const int vpu_ring_buffer_length = 16;

    for (auto input_bytes = 1; input_bytes < 48; ++input_bytes){

      std::list<std::tuple<int8_t, int8_t, int16_t>> args = { 
        std::tuple<int8_t, int8_t, int16_t>{1, 1, 1 },
        std::tuple<int8_t, int8_t, int16_t>{1, 0, 0 },
        std::tuple<int8_t, int8_t, int16_t>{0, 1, 0 },
        // std::tuple<int8_t, int8_t, int16_t>{-1, 1, -1 },
        // std::tuple<int8_t, int8_t, int16_t>{1, -1, -1 },
        std::tuple<int8_t, int8_t, int16_t>{-1, -1, 1 },
      };

      for (auto arg : args){
        int16_t expected_vD;
        int8_t kernel_fill, scratch_fill;
        std::tie(kernel_fill, scratch_fill, expected_vD) = arg;

        for (auto output_channel_count = 1; output_channel_count < 48; ++output_channel_count){
            
          int scratch_bytes = MatMulFn::get_scratch_size(input_bytes);
          int kernel_bytes = MatMulFn::get_kernel_size(input_bytes, output_channel_count);

          int8_t K[kernel_bytes];
          int8_t T[scratch_bytes];

          MatMulFn mm(output_channel_count, input_bytes, (int8_t *)K);

          std::fill_n(K, kernel_bytes, kernel_fill);
          std::fill_n(T, scratch_bytes, scratch_fill);


          int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

          for (auto ocg = 0; ocg < ocg_count; ++ocg){

            vpu_ring_buffer_t A;
            mm.aggregate_fn(&A, T, ocg);

            int c;
            if((ocg+1) * vpu_ring_buffer_length < output_channel_count)
              c = vpu_ring_buffer_length;
            else
              c = output_channel_count % vpu_ring_buffer_length;

            for (auto output_chan = 0; output_chan < c; ++output_chan){
              EXPECT_EQ(0, A.vR[output_chan]);
              EXPECT_EQ(scratch_bytes*expected_vD, A.vD[output_chan]);
            }
          }
        }
      }
    }
  }

  class Test_MatMulDirectFn: public ::testing::Test {};
  /*
    Simple test to verify memory accesses. 
  */
  TEST_F(Test_MatMulDirectFn, BasicTest) {

    const int vpu_bytes = 32;
    const int vpu_ring_buffer_length = 16;

    std::list<std::tuple<int8_t, int8_t, int16_t>> args = { 
      std::tuple<int8_t, int8_t, int16_t>{1, 1, 1 },
      std::tuple<int8_t, int8_t, int16_t>{1, 0, 0 },
      std::tuple<int8_t, int8_t, int16_t>{0, 1, 0 },
      // std::tuple<int8_t, int8_t, int16_t>{-1, 1, -1 },
      // std::tuple<int8_t, int8_t, int16_t>{1, -1, -1 },
      std::tuple<int8_t, int8_t, int16_t>{-1, -1, 1 },
    };

    for (auto arg : args){
      int16_t expected_vD;
      int8_t kernel_fill, scratch_fill;
      std::tie(kernel_fill, scratch_fill, expected_vD) = arg;

      for (auto x_height = 1; x_height <= 4; ++x_height){
        for (auto x_width = 1; x_width <= 4; ++x_width){
          for (auto x_channels = 32; x_channels <= 32*3; x_channels += 32){
            for (auto k_height = 1; k_height <= x_height; ++k_height){
              for (auto k_width = 1; k_width <= x_width; ++k_width){
                for (auto y_channels = 32; y_channels < 32*3; y_channels += 32){
                  
                  ImageParams X_params(x_height, x_width, x_channels, 8);
                  WindowGeometry K_params(k_height, k_width, 1, 1, 1, 1);

                  int8_t K[y_channels][k_height][k_width][x_channels];
                  int8_t T[x_height][x_width][x_channels];

                  int8_t * weights = (int8_t*)K; //todo we will switch to usnig the boggler

                  MatMulDirectFn mmd(X_params, K_params, weights);

                  std::fill_n((int8_t*)K, sizeof K, kernel_fill);
                  std::fill_n((int8_t*)T, x_height * x_width * x_channels, scratch_fill);

                  // std::cout <<"size " << x_height * x_width * x_channels << std::endl;

                  int ocg_count = (y_channels + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

                  for (auto x = 0; x < x_height - k_height + 1; ++x){
                    for (auto y = 0; y < x_width - k_width + 1; ++y){
                      for (auto ocg = 0; ocg < ocg_count; ++ocg){

                        vpu_ring_buffer_t A;
                        mmd.aggregate_fn(&A, (int8_t *)T, ocg);

                        for (auto output_chan = 0; output_chan < vpu_ring_buffer_length; ++output_chan){
                          EXPECT_EQ(0, A.vR[output_chan]);
                          EXPECT_EQ(k_width*k_height*x_channels*expected_vD, A.vD[output_chan]);
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

int main(int argc, char **argv) {


    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}