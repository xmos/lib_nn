#include "gtest/gtest.h"
#include <cstring>
#include <array>
#include <iostream>
#include <algorithm>

#include "MemCpyFn.hpp"
#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"
#include <tuple>
#include <list>

namespace {

  int pseudo_rand(int *seed){
    const int a = 1664525;
    const int c = 1013904223;
    *seed = (int)((long long)a * *seed + c);
    return *seed;
  }

  //TODO break these up into different files
  
  class Test_Im_to_col_valid: public ::testing::Test {};

  //TODO binary tests for Im_to_col_valid
  TEST_F(Test_Im_to_col_valid, BasicTest) {

    int seed = 42;

    for (auto x_height = 1; x_height <= 8; ++x_height){
      for (auto x_width = 1; x_width <= 8; ++x_width){
        for (auto x_channels = 4; x_channels <= 16; x_channels += 4){

          for (auto k_height = 1; k_height <= x_height; ++k_height){
            for (auto k_width = 1; k_width <= x_width; ++k_width){

              //the number of channels the kernel is going to be copying
              //TODO put input_ch_per_output in K.channels i.e. the output channels fo K
              for (int input_ch_per_output = 4; input_ch_per_output <= x_channels; input_ch_per_output += 4){

                for (auto k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation){
                  for (auto k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation){

                    for (auto k_h_stride = 1; k_h_stride <= 3; ++k_h_stride){ 
                      for (auto k_v_stride = 1; k_v_stride <= 3; ++k_v_stride){

                        int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                        int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

                        if (output_height <= 0 || output_width <= 0)
                          continue;
                          
                        ImageParams X(x_height, x_width, x_channels, 8); //8 bits per elemnet
                        WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                        Im_to_col_valid cpy(X, K, input_ch_per_output);

                        size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
                        int overread_bytes = cpy.get_overread_bytes(); //TODO add test that crashes when this is one less

                        int8_t T[scratch_bytes];
                        int8_t X_mem[x_width * x_height * x_channels + overread_bytes];

                        //The count of x channels from whick a memcopy could start 
                        int output_ch_starts = x_channels / input_ch_per_output;

                        for(int output_h = 0; output_h < output_height; ++output_h){
                          for(int output_w = 0 ; output_w < output_width; ++output_w){
                            for(int output_c = 0; output_c < output_ch_starts; output_c += 4){ //only test from aligned memory addresses

                              for(auto j = 0; j < sizeof X_mem; ++j)
                                X_mem[j] = (int8_t)pseudo_rand(&seed);

                              std::memset(T, 0x55, sizeof T);
                              
                              cpy.memcopy_fn(T, X_mem, output_h, output_w, output_c); 

                              int t_idx = 0;
                            
                              for(int kh = 0; kh < k_height; ++kh){
                                for(int kw = 0 ; kw < k_width; ++kw){
                                  for(int kc = 0 ; kc < input_ch_per_output; ++kc){ 
                                    int t = (int)T[t_idx++];
                                    //TODO use Aarons code here
                                    int x = (int)X_mem[(kh*k_v_dilation + k_v_stride*output_h)*x_channels*x_width + (kw*k_h_dilation+k_h_stride*output_w)*x_channels + kc+output_c];
                                    
                                    EXPECT_EQ(t, x);
                                  }
                                }
                              }
                              for(; t_idx < scratch_bytes; ++t_idx){
                                EXPECT_EQ(0, T[t_idx++]);
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

  class Test_Im_to_col_padded: public ::testing::Test {};

  TEST_F(Test_Im_to_col_padded, BasicTest) {

    int seed = 42;

    //TODO use above generator class

    for (int x_height = 1; x_height <= 4; ++x_height){
      for (int x_width = 1; x_width <= 4; ++x_width){
        for (int x_channels = 1; x_channels <= 8; x_channels += 1){
          for (int k_height = 1; k_height <= x_height; ++k_height){
            for (int k_width = 1; k_width <= x_width; ++k_width){
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation){
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation){
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride){ 
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride){
                      for (int input_ch_per_output = 1; input_ch_per_output <= x_channels; input_ch_per_output += 1){
                        for (int top_p=0;top_p <= 1;++top_p){
                          for (int bot_p=0;bot_p <= 1;++bot_p){
                            for (int left_p=0;left_p <= 1;++left_p){
                              for (int right_p=0;right_p <= 1;++right_p){

                                padding_t padding = {top_p, bot_p, left_p, right_p}; 
                          
                                int padded_x_height = padding.top + padding.bottom + x_height;
                                int padded_x_width = padding.left + padding.right + x_width;

                                int output_height = CONV2D_OUTPUT_LENGTH(padded_x_height, k_height, k_v_dilation, k_v_stride);
                                int output_width = CONV2D_OUTPUT_LENGTH(padded_x_width, k_width, k_h_dilation, k_h_stride);

                                if (output_height <= 0 || output_width <= 0)
                                  continue;

                                ImageParams X(x_height, x_width, x_channels, 8); 
                                WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                                int8_t pad_val = 0x55;
                                Im_to_col_padded cpy(X, K, padding, input_ch_per_output, pad_val);

                                size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
                                int overread_bytes = cpy.get_overread_bytes(); //TODO add test that crashes when this is one less

                                int8_t T[scratch_bytes];
                                int8_t X_mem[x_height][x_width][x_channels];

                                //create an explicitly padded version of X
                                int8_t X_mem_padded[padded_x_height][padded_x_width][x_channels];
                                
                                for (auto i=0;i<sizeof X_mem; ++i)
                                  ((int8_t*)X_mem )[i]= (int8_t)pseudo_rand(&seed);
                                
                                std::memset(X_mem_padded, pad_val, sizeof X_mem_padded);

                                for(auto h=0;h<x_height;h++){
                                  for(auto w=0;w<x_width;w++){
                                    for(auto c=0;c<x_channels;c++){
                                      X_mem_padded[h + padding.top][w + padding.left][c] =  X_mem[h][w][c];
                                    }
                                  }
                                }

                                int8_t X_mem_with_overread[sizeof X_mem + overread_bytes];

                                std::memcpy(X_mem_with_overread, (int8_t*)X_mem, sizeof X_mem);
                                for(auto j = sizeof X_mem; j < sizeof X_mem_with_overread; ++j)
                                  X_mem_with_overread[j] = (int8_t)pseudo_rand(&seed);

                                int output_ch_starts = x_channels / input_ch_per_output;

                                for(int output_h = 0; output_h < output_height; ++output_h){
                                  for(int output_w = 0 ; output_w < output_width; ++output_w){
                                    for(int output_c = 0; output_c < output_ch_starts; ++output_c){ //only test from aligned memory addresses
                                    
                                      std::memset(T, 0xaa, sizeof T);

                                      cpy.memcopy_fn(T, X_mem_with_overread, output_h, output_w, output_c); 

                                      int t_idx = 0;
                                    
                                      for(int kh = 0; kh < k_height; ++kh){
                                        for(int kw = 0 ; kw < k_width; ++kw){
                                          for(int kc = 0 ; kc < input_ch_per_output; ++kc){ 
                                            int t = (int)T[t_idx++];

                                            //TODO use Aarons code here
                                            int x = (int)X_mem_padded[kh*k_v_dilation + k_v_stride*output_h][kw*k_h_dilation+k_h_stride*output_w][kc+output_c];
                                            EXPECT_EQ(t, x);
                                          }
                                        }
                                      }
                                      for(; t_idx < scratch_bytes; ++t_idx){
                                        EXPECT_EQ(0, T[t_idx++]);
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

  class Test_DerefInputFn: public ::testing::Test {};

  TEST_F(Test_DerefInputFn, BasicTest) {

    int seed = 69;
    
    int k_h_dilation = 1;
    int k_v_dilation = 1;

    for (int x_height = 1; x_height <= 10; ++x_height){
      for (int x_width = 1; x_width <= 10; ++x_width){
        for (int x_channels = 1; x_channels <= 8; x_channels += 1){
          for (int k_height = 1; k_height <= x_height; ++k_height){
            for (int k_width = 1; k_width <= x_width; ++k_width){
              for (int k_h_stride = 1; k_h_stride <= 4; ++k_h_stride){
                for (int k_v_stride = 1; k_v_stride <= 4; ++k_v_stride){
                  
                  int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                  int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

                  if (output_height <= 0 || output_width <= 0)
                    continue;
                    
                  ImageParams X(x_height, x_width, x_channels, 8); 
                  WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                  DerefInputFn deref(X, K);
                  
                  int8_t X_mem[x_height][x_width][x_channels];

                  for(auto j = 0; j < sizeof X_mem; ++j)
                    ((int8_t*)X_mem)[j] = (int8_t)pseudo_rand(&seed);

                  for(auto h = 0; h < output_height; ++h){
                    for(auto w = 0 ; w < output_width; ++w){
                      for(auto c = 0 ; c < x_channels; ++c){
                        int8_t * p = deref.memcopy_fn(0, (int8_t*)X_mem, h, w, c);
                        int x = (int)X_mem[k_v_stride*h][k_h_stride*w][c];
                        EXPECT_EQ((int)*p, x);
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


  //////////////////////////////////////////////////////////////////////////////////////////


  class Test_SimpleMatMulFn: public ::testing::Test {};
  /*
    Simple test to verify memory accesses
  */
  TEST_F(Test_SimpleMatMulFn, BasicTest) {
    // const int vpu_bytes = 32;
    const int vpu_ring_buffer_length = 16;

    for (auto input_bytes = 1; input_bytes < 48; ++input_bytes){

      std::list<std::tuple<int8_t, int8_t>> args = { 
        std::tuple<int8_t, int8_t>{1, 1 },
        std::tuple<int8_t, int8_t>{1, 0},
        std::tuple<int8_t, int8_t>{0, 1 },
        std::tuple<int8_t, int8_t>{-1, 1},
        std::tuple<int8_t, int8_t>{1, -1},
        std::tuple<int8_t, int8_t>{-1, -1},
      };

      for (auto arg : args){
        int8_t kernel_fill, scratch_fill;
        std::tie(kernel_fill, scratch_fill) = arg;

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

            int32_t v;
            ((int16_t *)&v)[0] = A.vD[output_chan];
            ((int16_t *)&v)[1] = A.vR[output_chan];

              EXPECT_EQ(scratch_bytes*(kernel_fill*scratch_fill), v);
            }
          }
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

    int seed = 69;

    for (auto input_bytes = 1; input_bytes < 128; ++input_bytes){

      for (auto output_channel_count = 1; output_channel_count < 48; ++output_channel_count){
          
        int k_height = 1;
        int k_width = 1; //to make things easy

        std::array<int, 4> shape = {output_channel_count, k_height, k_width, input_bytes};
        int8_t raw_weights[output_channel_count][k_height][k_width][input_bytes];
        assert(sizeof raw_weights == input_bytes*output_channel_count);

        for(auto j = 0; j < sizeof raw_weights; ++j)
          ((int8_t*)raw_weights)[j] = (int8_t)pseudo_rand(&seed);

        int scratch_bytes = MatMulFn::get_scratch_size(input_bytes);

        int8_t* reordered_weights;
        int8_t** final_load_locations;
        int kernel_bytes;

        int8_t pad_val = (int8_t)pseudo_rand(&seed);

        std::tie(reordered_weights, final_load_locations, kernel_bytes) = 
          MatMulFn::reorder_kernel_weights( (int8_t* )raw_weights, shape, 8, pad_val) ;
        
        int8_t T[scratch_bytes];

        for(auto j = 0; j < sizeof T; ++j)
          T[j] = (int8_t)pseudo_rand(&seed);

        int accu_modifier[output_channel_count]; //=0

        //TODO make this into an int8 specific function
        for (int i=0;i<output_channel_count;i++){
          int8_t * final_load_location = final_load_locations[i];
          
          int s = 0;
          int channel_overlap_start = input_bytes%vpu_bytes;

          if (channel_overlap_start){

            for(int j=channel_overlap_start;j<vpu_bytes;j++){
              s += (int)(final_load_location[j]) * T[scratch_bytes - vpu_bytes + j]; 
            }
            
          }
          accu_modifier[i] = s;
        }

        MatMulFn mm(output_channel_count, input_bytes, reordered_weights);
        int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;
        
        for (auto ocg = 0; ocg < ocg_count; ++ocg){

          vpu_ring_buffer_t A;
          mm.aggregate_fn(&A, T, ocg);

          int chs_in_group = std::min(output_channel_count - output_channel_count *ocg , vpu_ring_buffer_length);

          for (auto output_chan = 0; output_chan < chs_in_group; ++output_chan){

            int actual_output_channel = output_chan + ocg * vpu_ring_buffer_length;

            int expected_sum = 0;
            for(int b = 0; b < input_bytes; b++)
              expected_sum += ((int)raw_weights[actual_output_channel][0][0][b] * (int)T[b]);
            
            int32_t v;
            ((int16_t *)&v)[0] = A.vD[output_chan];
            ((int16_t *)&v)[1] = A.vR[output_chan];

            EXPECT_EQ( v - accu_modifier[actual_output_channel], expected_sum);
          }
        }

        delete reordered_weights;
        delete final_load_locations;
      }

    }
  }

  class Test_Simple_MatMulDirectFn: public ::testing::Test {};
  /*
    Simple test to verify memory accesses. 
  */
  TEST_F(Test_Simple_MatMulDirectFn, BasicTest) {

    const int vpu_ring_buffer_length = 16;

    std::list<std::tuple<int8_t, int8_t>> args = { 
      std::tuple<int8_t, int8_t>{1, 1 },
      std::tuple<int8_t, int8_t>{1, 0},
      std::tuple<int8_t, int8_t>{0, 1 },
      std::tuple<int8_t, int8_t>{-1, 1},
      std::tuple<int8_t, int8_t>{1, -1},
      std::tuple<int8_t, int8_t>{-1, -1},
    };

    for (auto arg : args){
      int8_t kernel_fill, scratch_fill;
      std::tie(kernel_fill, scratch_fill) = arg;

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

                  MatMulDirectFn mmd(X_params, K_params, x_channels, weights);

                  std::fill_n((int8_t*)K, sizeof K, kernel_fill);
                  std::fill_n((int8_t*)T, x_height * x_width * x_channels, scratch_fill);

                  int ocg_count = (y_channels + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

                  for (auto x = 0; x < x_height - k_height + 1; ++x){
                    for (auto y = 0; y < x_width - k_width + 1; ++y){
                      for (auto ocg = 0; ocg < ocg_count; ++ocg){

                        vpu_ring_buffer_t A;
                        mmd.aggregate_fn(&A, (int8_t *)T, ocg);

                        for (auto output_chan = 0; output_chan < vpu_ring_buffer_length; ++output_chan){

                          int32_t v;
                          ((int16_t *)&v)[0] = A.vD[output_chan];
                          ((int16_t *)&v)[1] = A.vR[output_chan];

                          EXPECT_EQ(k_width*k_height*x_channels*(kernel_fill*scratch_fill), v);
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


  class Test_MatMulDirectFn: public ::testing::Test {};
  /*
    Simple test to verify memory accesses. 
  */
  TEST_F(Test_MatMulDirectFn, BasicTest) {
    // const int vpu_bytes = 32;
    const int vpu_ring_buffer_length = 16;

    int seed = 69;
    
    int k_h_stride = 1;
    int k_v_stride = 1;

    //TODO replace 16 and 32
    for (int x_height = 1; x_height <= 6; ++x_height){
      for (int x_width = 1; x_width <= 6; ++x_width){
        for (int x_channels = 32; x_channels <= 32*3; x_channels += 32){
          for (int k_height = 1; k_height <= x_height; ++k_height){
            for (int k_width = 1; k_width <= x_width; ++k_width){
              for (int k_h_dilation = 1; k_h_dilation <= 4; ++k_h_dilation){
                for (int k_v_dilation = 1; k_v_dilation <= 4; ++k_v_dilation){
                  for (int output_channels = 16; output_channels <= 16*3; output_channels += 16){
                    for (int input_ch_per_output = x_channels; input_ch_per_output <= x_channels; input_ch_per_output += 32){
                  
                      int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                      int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

                      if (output_height <= 0 || output_width <= 0)
                        continue;
                        
                      // std::cout << "x_height: " << x_height
                      //           << " x_width: " << x_width
                      //           << " x_channels: " << x_channels
                      //           << " k_height: " << k_height
                      //           << " k_width: " << k_width
                      //           << " k_h_dilation: " << k_h_dilation
                      //           << " k_v_dilation: " << k_v_dilation
                      //           << " output_channels: " << output_channels
                      //           << " input_ch_per_output: " << input_ch_per_output 
                      //           << std::endl;

                      ImageParams X(x_height, x_width, x_channels, 8); 
                      WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                      std::array<int, 4> shape = {output_channels, k_height, k_width, x_channels};
                      int8_t raw_weights[output_channels][k_height][k_width][x_channels];

                      for(auto j = 0; j < sizeof raw_weights; ++j)
                        ((int8_t*)raw_weights)[j] = (int8_t)pseudo_rand(&seed);

                      int8_t X_mem[x_height][x_width][x_channels];

                      for(auto j = 0; j < sizeof X_mem; ++j)
                        ((int8_t*)X_mem)[j] = (int8_t)pseudo_rand(&seed);

                      int8_t* reordered_weights;
                      int8_t** final_load_locations;
                      int kernel_bytes;

                      int8_t pad_val = (int8_t)pseudo_rand(&seed); //this should be unused in this case

                      std::tie(reordered_weights, final_load_locations, kernel_bytes) = 
                        MatMulFn::reorder_kernel_weights( (int8_t* )raw_weights, shape, 8, pad_val) ;

                      MatMulDirectFn mmd(X, K, input_ch_per_output, reordered_weights);
                      
                      int ocg_count = (output_channels + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;
                      
                      for (auto ocg = 0; ocg < ocg_count; ++ocg){

                        vpu_ring_buffer_t A;
                        // printf("start %p size:%d\n",  (int8_t*)X_mem, sizeof X_mem);
                        mmd.aggregate_fn(&A, (int8_t*)X_mem, ocg);

                        int chs_in_group = std::min(output_channels - vpu_ring_buffer_length *ocg , vpu_ring_buffer_length);

                        for (auto output_chan = 0; output_chan < chs_in_group; ++output_chan){

                          int actual_output_channel = output_chan + ocg * vpu_ring_buffer_length;

                          int expected_sum = 0;

                          for(auto h = 0; h < k_height; ++h){
                            for(auto w = 0 ; w < k_width; ++w){
                              for(auto c = 0 ; c < input_ch_per_output; ++c){
                                // std::cout <<"h: "<<h << " w: "<<w << " c: "<<c<<std::endl;
                                int x = (int)X_mem[k_v_dilation*h][k_h_dilation*w][c];
                                int t = raw_weights[actual_output_channel][h][w][c];
                                expected_sum += x*t;
                              }
                            }
                          }

                          int32_t v;
                          ((int16_t *)&v)[0] = A.vD[output_chan];
                          ((int16_t *)&v)[1] = A.vR[output_chan];
                          // std::cout << actual_output_channel<< " " << v << " " << expected_sum << std::endl;
                          EXPECT_EQ(v, expected_sum);
                        }
                      }

                      delete reordered_weights;
                      delete final_load_locations;

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
            
            int8_t* reordered_weights;
            int8_t** final_load_locations;
            int kernel_bytes;

            std::tie(reordered_weights, final_load_locations, kernel_bytes) = 
              MatMulFn::reorder_kernel_weights((int8_t *)raw_weights, shape, bits_per_element, 0);

            delete reordered_weights;
            delete final_load_locations;

          }
        }
      }
    }
  }

 class Test_OT_int8: public ::testing::Test {};

  TEST_F(Test_OT_int8, BasicTest) {
    const int vpu_ring_buffer_length = VPU_INT16_EPV;


    for(int output_ch_count = 1; output_ch_count < 32; ++output_ch_count){



      std::vector<float> f_biases; //[output_ch_count];
      std::vector<float> f_multipliers; //[output_ch_count];
      int32_t accu_max = INT16_MAX;
      int32_t accu_min = INT16_MIN;

      int t = ((output_ch_count + vpu_ring_buffer_length-1)/vpu_ring_buffer_length)*vpu_ring_buffer_length;

      int16_t accu_modifier[t]; //this comes from the aggregate fn

      QuantisationParams qp = OTBinary_int8::quantise_activation(f_multipliers, f_biases, accu_min, accu_max);

      OTBinary_int8 ot((int32_t)output_ch_count, &qp.otv, qp.biases.data(), 
        qp.multipliers.data(), (int16_t*)accu_modifier);

      int8_t Y[output_ch_count];

      int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

// int8_t * OTBinary_int8::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)

      int8_t *y = (int8_t*)Y;
      for (int ocg=0; ocg < ocg_count; ++ocg){

        int chs_in_group = std::min(output_ch_count - output_ch_count * ocg , vpu_ring_buffer_length);

        vpu_ring_buffer_t A;

        //fill A with random value between accu_max and accu_min

        //then add accu_modifier to each channel

        y = ot.output_transform_fn(y, &A, ocg);


      }

    }




  }


  //end
}

int main(int argc, char **argv) {


    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}