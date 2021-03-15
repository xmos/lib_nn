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

  //TODO breal these up into different files

  class Test_Im_to_col_valid: public ::testing::Test {};

  TEST_F(Test_Im_to_col_valid, BasicTest) {

    int seed = 42;

    //TODO put this in a generator class
    for (auto x_height = 1; x_height <= 6; ++x_height){
      for (auto x_width = 1; x_width <= 6; ++x_width){
        for (auto x_channels = 4; x_channels <= 36; x_channels += 4){
          for (auto k_height = 1; k_height <= x_height; ++k_height){
            for (auto k_width = 1; k_width <= x_width; ++k_width){
              for (auto k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation){
                for (auto k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation){
                  for (auto k_h_stride = 1; k_h_stride <= 1; ++k_h_stride){ //TODO test this
                    for (auto k_v_stride = 1; k_v_stride <= 1; ++k_v_stride){
                      
                      ImageParams X(x_height, x_width, x_channels, 8); 
                      WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                      Im_to_col_valid cpy(X, K);

                      size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
                      int overread_bytes = cpy.get_overread_bytes(); //TODO add test that crashes when this is one less

                      int8_t T[scratch_bytes];
                      int8_t X_mem[x_width * x_height * x_channels + overread_bytes];

                      int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                      int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

                      for(auto h = 0; h < output_height; ++h){
                        for(auto w = 0 ; w < output_width; ++w){

                          for(auto j = 0; j < x_width * x_height * x_channels + overread_bytes; ++j)
                            X_mem[j] = (int8_t)pseudo_rand(&seed);

                          std::memset(T, 0x55, sizeof T);
                          
                          cpy.memcopy_fn(T, X_mem, h, w, 0);

                          int j = 0;

                          int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                          int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);
                          
                          for(int kh = 0; kh < k_height; ++kh){
                            for(int kw = 0 ; kw < k_width; ++kw){
                              for(int c = 0 ; c < x_channels; ++c){

                                EXPECT_EQ((int)T[j++], (int)X_mem[(kh*k_v_dilation +h)*x_channels*x_width + (kw*k_h_dilation+w)*x_channels + c]);
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
                }
              }
            }
          }
        }
      }
    }
  }

  class Test_DerefInputFn: public ::testing::Test {};

  TEST_F(Test_DerefInputFn, BasicTest) {

    int seed = 42;

    //TODO use above generator class
    //TODO add strides and dilations etc
    int k_height= 1, k_v_dilation=1, k_v_stride=1;
    int k_width=1, k_h_dilation=1, k_h_stride=1;
    for (auto x_height = 1; x_height <= 6; ++x_height){
      for (auto x_width = 1; x_width <= 6; ++x_width){
        for (auto x_channels = 4; x_channels <= 36; x_channels += 4){

          ImageParams X(x_height, x_width, x_channels, 8); 
          WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

          DerefInputFn deref(X, K);
          
          int8_t X_mem[x_width * x_height * x_channels ];
          for(auto j = 0; j < x_width * x_height * x_channels; ++j)
            X_mem[j] = (int8_t)pseudo_rand(&seed);

          int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
          int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

          for(auto h = 0; h < output_height; ++h){
            for(auto w = 0 ; w < output_width; ++w){
              
              int8_t * p = deref.memcopy_fn(0, X_mem, h, w, 0);

              EXPECT_EQ((int)*p, (int)X_mem[h*x_channels*x_width + w*x_channels]);
 
            }
          }
        }
      }
    }
  }

  class Test_Im_to_col_padded: public ::testing::Test {};

  TEST_F(Test_Im_to_col_padded, BasicTest) {

    int seed = 42;

    //TODO use above generator class
    auto k_h_stride = 1;
    auto k_v_stride = 1;


    for (auto x_height = 1; x_height <= 6; ++x_height){
      for (auto x_width = 1; x_width <= 6; ++x_width){
        for (auto x_channels = 1; x_channels <= 36; x_channels += 1){
          for (auto k_height = 1; k_height <= x_height; ++k_height){
            for (auto k_width = 1; k_width <= x_width; ++k_width){
              for (auto k_h_dilation = 1; k_h_dilation <= 1; ++k_h_dilation){
                for (auto k_v_dilation = 1; k_v_dilation <= 1; ++k_v_dilation){

                  padding_t padding = {0, 0, 0, 0}; //TODO add this to the generator

                  // std::cout << " x_height:" << x_height
                  //  << " x_width:" << x_width
                  //   << " x_height:" << x_height
                  //    << " x_channels:" << x_channels
                  //     << " k_height:" << k_height
                  //      << " k_width:" << k_width
                  //       << " k_h_dilation:" << k_h_dilation
                  //        << " k_v_dilation:" << k_v_dilation
                  //          <<std::endl;

                  ImageParams X(x_height, x_width, x_channels, 8); 
                  WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                  Im_to_col_padded cpy(X, K, padding);

                  size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
                  int overread_bytes = cpy.get_overread_bytes(); //TODO add test that crashes when this is one less

                  int8_t T[scratch_bytes];
                  int8_t X_mem[x_width * x_height * x_channels + overread_bytes];

                  int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                  int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);
                  if(output_height ==0 || output_width == 0){
                    std::cout << "\tdid nothing" << std::endl;
                  }
                  for(auto h = 0; h < output_height; ++h){
                    for(auto w = 0 ; w < output_width; ++w){

                      for(auto j = 0; j < x_width * x_height * x_channels + overread_bytes; ++j)
                        X_mem[j] = (int8_t)pseudo_rand(&seed);

                      std::memset(T, 0x55, sizeof T);
                      
                      cpy.memcopy_fn(T, X_mem, h, w, 0);

                      // int j = 0;
                      // for(int kh = 0; kh < k_height; ++kh){
                      //   for(int kw = 0 ; kw < k_width; ++kw){
                      //     for(int c = 0 ; c < x_channels; ++c){
                      //       EXPECT_EQ((int)T[j++], (int)X_mem[(kh*k_v_dilation +h)*x_channels*x_width + (kw*k_h_dilation+w)*x_channels + c]);
                      //     }
                      //   }
                      // }
                      // for(; j < scratch_bytes; ++j){
                      //   EXPECT_EQ(0, T[j++]);
                      // }
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

  //////////////////////////////////////////////////////////////////////////////////////////


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

  class Test_Kernel_Reordering: public ::testing::Test {};

  TEST_F(Test_Kernel_Reordering, BasicTest) {

    for (auto x_channels = 1; x_channels <= 6; ++x_channels){
      for (auto k_height = 1; k_height <= 6; ++k_height){
        for (auto k_width = 1; k_width <= 6; ++k_width){
          for (auto y_channels = 1; y_channels <= 6; ++y_channels){

            int8_t raw_weights[y_channels][k_height][k_width][x_channels];

            std::array<int, 4> shape = {y_channels, k_height, k_width, x_channels};
            int bits_per_element = 8;

            memset(raw_weights, 0, sizeof raw_weights); 

            //should return the size and pointers to final vpu loads
            int8_t * reordered_weights = 
              MatMulFn::reorder_kernel_weights((int8_t *)raw_weights, shape, bits_per_element, 0);

            //check that all values are 0
            
            for (auto i = 0; i < sizeof raw_weights; ++i){
              EXPECT_EQ(0, reordered_weights[i]);
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