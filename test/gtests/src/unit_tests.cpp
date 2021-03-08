#include "gtest/gtest.h"
#include <cstring>

#include "MemCpyFn.hpp"

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

      int overread_bytes = 0; cpy.get_overread_bytes();

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

  class Test_Im_to_col_padded: public ::testing::Test {};

  TEST_F(Test_Im_to_col_padded, BasicTest) {

    int x_width = 12;
    int x_height = 12;
    int x_channels = 3;

    int k_width = 3;
    int k_height = 3;
    int k_channels = 8;

    padding_t padding = {1, 1, 1, 1};

    ImageParams X(x_width, x_height, x_channels, 8); 
    WindowGeometry K (k_width, k_height, 1, 1, 1, 1);
    ImageParams Y(X, K, k_channels);

    Im_to_col_padded cpy(X, Y, K, padding);
    size_t scratch_bytes = cpy.get_scratch_bytes();

    int8_t T[scratch_bytes];
    int8_t X_mem[x_width][x_height][x_channels];

    for(int h=0;h<x_height-k_width; ++h){
      for(int w=0;w<x_width-k_height; ++w){

        //todo random init X_mem

        cpy.memcopy_fn(T, (int8_t*)X_mem, h, w);

        //todo verify the correct bytes were copied

      }
    }

    EXPECT_EQ(0, 0);
  }

  struct Conv2DParams{

    int x_height;
    int x_width;
    int x_channels;

    int k_height;
    int k_width;
    int k_channels;

    int k_dilation_h;
    int k_dilation_v;

    int k_stride_h;
    int k_stride_v;

  };

  class Conv2DIterator {

    int cur_

    public:
    Conv2DIterator(Conv2DParams &param_min, Conv2DParams &param_max);

  }

  Conv2DIterator::Conv2DIterator(Conv2DParams &param_min, Conv2DParams &param_max){

  }
}

int main(int argc, char **argv) {


    // ::testing::InitGoogleTest(&argc, argv);
    // return RUN_ALL_TESTS();
}