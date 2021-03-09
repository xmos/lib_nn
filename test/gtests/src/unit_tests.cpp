#include "gtest/gtest.h"
#include <cstring>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"

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

  TEST_F(Test_MatMulFn, BasicTest) {

    int k_width = 3;
    int k_height = 3;
    int input_channels = 8;

    int output_channel_count = 4;

    int8_t K[output_channel_count][k_height][k_width][input_channels];
    size_t bytes_per_kernel_channel = k_height*k_width*input_channels;
    
    size_t scratch_bytes = bytes_per_kernel_channel + 32;
    int8_t T[scratch_bytes];

    int output_channel_groups = (output_channel_count  + 16 - 1)/ 16;

    int8_t * weights = 0;//MatMulFn::boggle(K);

    MatMulFn mm(output_channel_count, bytes_per_kernel_channel, weights);
    int seed = 69;
    for (auto ocg =0; ocg < output_channel_groups; ++ocg){
      vpu_ring_buffer_t A;

      for(auto j = 0; j < scratch_bytes; ++j)
        T[j] = (int8_t)pseudo_rand(&seed);

      // mm.aggregate_fn(&A, T, ocg);

      int ocg_chanel_count = 16;

      for (auto output_chan = 0; output_chan < 16; ++output_chan){
        
        //reinterpret K as a 1D array
        int8_t * k = reinterpret_cast<int8_t*>(&K);

        //product it with T
        int32_t agg_sum = 0;
        for(auto i=0; i<bytes_per_kernel_channel; ++i)
          // agg_sum = vpu_prod(agg_sum, k[i], T[i]);
        
        //deal with the over run with the next output channel
        if(1){
          for(auto bytes_per_kernel_channel=0; i<scratch_bytes; ++i){
            // agg_sum = vpu_prod(agg_sum, k[i], T[i]);
          }
        } else {
        for(auto bytes_per_kernel_channel=0; i<scratch_bytes; ++i){
          // agg_sum = vpu_prod(agg_sum, k[i], T[i]);
        }
        }
      }


    }

    EXPECT_EQ(0, 0);
  }


}

int main(int argc, char **argv) {


    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}