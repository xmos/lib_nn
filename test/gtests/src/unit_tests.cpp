#include "gtest/gtest.h"

#include <cstring>
#include <cmath>
#include <array>
#include <iostream>
#include <algorithm>
#include <tuple>
#include <list>
#include <vector>

#include "Filter2D.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"

namespace {

  int pseudo_rand(int *seed){
    const int a = 1664525;
    const int c = 1013904223;
    *seed = (int)((long long)a * *seed + c);
    return *seed;
  }
  //TODO break these up into different files
  
  class Test_ImToColValid: public ::testing::Test {};

  //TODO binary tests for ImToColValid
  TEST_F(Test_ImToColValid, BasicTest) {

    int seed = 42;

    for (int x_height = 1; x_height <= 8; ++x_height){
      for (int x_width = 1; x_width <= 8; ++x_width){
        for (int x_channels = 4; x_channels <= 16; x_channels += 4){

          for (int k_height = 1; k_height <= x_height; ++k_height){
            for (int k_width = 1; k_width <= x_width; ++k_width){

              //the number of channels the kernel is going to be copying
              //TODO put input_ch_per_output in K.channels i.e. the output channels fo K
              for (int input_ch_per_output = 4; input_ch_per_output <= x_channels; input_ch_per_output += 4){

                for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation){
                  for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation){

                    for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride){ 
                      for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride){

                        int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                        int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

                        if (output_height <= 0 || output_width <= 0)
                          continue;
                          
                        ImageParams X(x_height, x_width, x_channels, 8); //8 bits per elemnet
                        WindowGeometry K (k_height, k_width, k_h_stride, k_v_stride, k_h_dilation, k_v_dilation);

                        ImToColValid::Params p(X, K, input_ch_per_output);
                        ImToColValid cpy(&p);

                        size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
                        int overread_bytes = cpy.get_overread_bytes(); //TODO add test that crashes when this is one less

                        int8_t T[scratch_bytes];
                        int8_t X_mem[x_width * x_height * x_channels + overread_bytes];

                        //The count of x channels from whick a memcopy could start 
                        int output_ch_starts = x_channels / input_ch_per_output;

                        for(int output_h = 0; output_h < output_height; ++output_h){
                          for(int output_w = 0 ; output_w < output_width; ++output_w){
                            for(int output_c = 0; output_c < output_ch_starts; output_c += 4){ //only test from aligned memory addresses

                              for(int j = 0; j < sizeof X_mem; ++j)
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

  class Test_ImToColPadded: public ::testing::Test {};

  TEST_F(Test_ImToColPadded, BasicTest) {

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

                                ImToColPadded::Params p(X, K, padding, input_ch_per_output, pad_val);
                                ImToColPadded cpy(&p);

                                size_t scratch_bytes = cpy.get_scratch_bytes(); //TODO add test that crashes when this is one less
                                int overread_bytes = cpy.get_overread_bytes(); //TODO add test that crashes when this is one less

                                int8_t T[scratch_bytes];
                                int8_t X_mem[x_height][x_width][x_channels];

                                //create an explicitly padded version of X
                                int8_t X_mem_padded[padded_x_height][padded_x_width][x_channels];
                                
                                for (int i=0;i<sizeof X_mem; ++i)
                                  ((int8_t*)X_mem )[i]= (int8_t)pseudo_rand(&seed);
                                
                                std::memset(X_mem_padded, pad_val, sizeof X_mem_padded);

                                for(int h=0;h<x_height;h++){
                                  for(int w=0;w<x_width;w++){
                                    for(int c=0;c<x_channels;c++){
                                      X_mem_padded[h + padding.top][w + padding.left][c] =  X_mem[h][w][c];
                                    }
                                  }
                                }

                                int8_t X_mem_with_overread[sizeof X_mem + overread_bytes];

                                std::memcpy(X_mem_with_overread, (int8_t*)X_mem, sizeof X_mem);
                                for(int j = sizeof X_mem; j < sizeof X_mem_with_overread; ++j)
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

                  DerefInputFn::Params p(X, K);
                  DerefInputFn deref(&p);
                  
                  int8_t X_mem[x_height][x_width][x_channels];

                  for(int j = 0; j < sizeof X_mem; ++j)
                    ((int8_t*)X_mem)[j] = (int8_t)pseudo_rand(&seed);

                  for(int h = 0; h < output_height; ++h){
                    for(int w = 0 ; w < output_width; ++w){
                      for(int c = 0 ; c < x_channels; ++c){
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


  class Test_SimpleMatMulInt8: public ::testing::Test {};
  /*
    Simple test to verify memory accesses
  */
  TEST_F(Test_SimpleMatMulInt8, BasicTest) {
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

        for (int output_channel_count = 1; output_channel_count < 48; ++output_channel_count){
            
          int scratch_bytes = MatMulInt8::get_scratch_size(input_bytes);
          int kernel_bytes = MatMulInt8::get_kernel_size(input_bytes, output_channel_count);

          int8_t K[kernel_bytes];
          int8_t T[scratch_bytes];

          MatMulInt8::Params p(output_channel_count, input_bytes, (int8_t *)K);
          MatMulInt8 mm(&p);

          std::fill_n(K, kernel_bytes, kernel_fill);
          std::fill_n(T, scratch_bytes, scratch_fill);

          int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

          for (int ocg = 0; ocg < ocg_count; ++ocg){

            vpu_ring_buffer_t A;
            mm.aggregate_fn(&A, T, ocg);

            int c;
            if((ocg+1) * vpu_ring_buffer_length < output_channel_count)
              c = vpu_ring_buffer_length;
            else
              c = output_channel_count % vpu_ring_buffer_length;

            for (int output_chan = 0; output_chan < c; ++output_chan){

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

  class Test_MatMulInt8: public ::testing::Test {};
  /*
    Simple test to verify memory accesses
  */
  TEST_F(Test_MatMulInt8, BasicTest) {
    const int vpu_bytes = 32;
    const int vpu_ring_buffer_length = 16;

    int seed = 69;

    for (int input_bytes = 1; input_bytes < 128; ++input_bytes){

      for (int output_channel_count = 1; output_channel_count < 48; ++output_channel_count){
          
        int k_height = 1;
        int k_width = 1; //to make things easy

        std::array<int, 4> shape = {output_channel_count, k_height, k_width, input_bytes};
        int8_t raw_weights[output_channel_count][k_height][k_width][input_bytes];
        assert(sizeof raw_weights == input_bytes*output_channel_count);

        for(auto j = 0; j < sizeof raw_weights; ++j)
          ((int8_t*)raw_weights)[j] = (int8_t)pseudo_rand(&seed);

        int scratch_bytes = MatMulInt8::get_scratch_size(input_bytes);

        // int8_t* reordered_weights;
        // int8_t** final_load_locations;
        // int kernel_bytes;

        int8_t pad_val = (int8_t)pseudo_rand(&seed);

        // std::tie(reordered_weights, final_load_locations, kernel_bytes) = 
        Conv2dReorderedWeights rw = 
          MatMulInt8::reorder_kernel_weights( (int8_t* )raw_weights, shape, 8, pad_val) ;
        
        int8_t T[scratch_bytes];

        for(int j = 0; j < sizeof T; ++j)
          T[j] = (int8_t)pseudo_rand(&seed);

        int accu_modifier[output_channel_count]; //=0

        //TODO make this into an int8 specific function
        for (int i=0;i<output_channel_count;i++){
          int idx = rw.final_vpu_load_addresses[i];
          // int8_t * final_load_location = final_load_locations[i];
          
          int s = 0;
          int channel_overlap_start = input_bytes%vpu_bytes;

          if (channel_overlap_start){

            for(int j=channel_overlap_start;j<vpu_bytes;j++){
              // s += (int)(final_load_location[j]) * T[scratch_bytes - vpu_bytes + j]; 
              s += (int)(rw.weights[idx+j]) * T[scratch_bytes - vpu_bytes + j]; 
            }
            
          }
          accu_modifier[i] = s;
        }

        MatMulInt8::Params p(output_channel_count, input_bytes, rw.weights.data());
        MatMulInt8 mm(&p);
        int ocg_count = (output_channel_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;
        
        for (int ocg = 0; ocg < ocg_count; ++ocg){

          vpu_ring_buffer_t A;
          mm.aggregate_fn(&A, T, ocg);

          int chs_in_group = std::min(output_channel_count - output_channel_count *ocg , vpu_ring_buffer_length);

          for (int output_chan = 0; output_chan < chs_in_group; ++output_chan){

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

        // delete reordered_weights;
        // delete final_load_locations;
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

      for (int x_height = 1; x_height <= 4; ++x_height){
        for (int x_width = 1; x_width <= 4; ++x_width){
          for (int x_channels = 32; x_channels <= 32*3; x_channels += 32){
            for (int k_height = 1; k_height <= x_height; ++k_height){
              for (int k_width = 1; k_width <= x_width; ++k_width){
                for (int y_channels = 32; y_channels < 32*3; y_channels += 32){
                  
                  ImageParams X_params(x_height, x_width, x_channels, 8);
                  WindowGeometry K_params(k_height, k_width, 1, 1, 1, 1);

                  int8_t K[y_channels][k_height][k_width][x_channels];
                  int8_t T[x_height][x_width][x_channels];

                  int8_t * weights = (int8_t*)K; //todo we will switch to usnig the boggler


                  MatMulDirectFn::Params p(X_params, K_params, x_channels, weights);
                  MatMulDirectFn mmd(&p);

                  std::fill_n((int8_t*)K, sizeof K, kernel_fill);
                  std::fill_n((int8_t*)T, x_height * x_width * x_channels, scratch_fill);

                  int ocg_count = (y_channels + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

                  for (int x = 0; x < x_height - k_height + 1; ++x){
                    for (int y = 0; y < x_width - k_width + 1; ++y){
                      for (int ocg = 0; ocg < ocg_count; ++ocg){

                        vpu_ring_buffer_t A;
                        mmd.aggregate_fn(&A, (int8_t *)T, ocg);

                        for (int output_chan = 0; output_chan < vpu_ring_buffer_length; ++output_chan){

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

                      for(int j = 0; j < sizeof raw_weights; ++j)
                        ((int8_t*)raw_weights)[j] = (int8_t)pseudo_rand(&seed);

                      int8_t X_mem[x_height][x_width][x_channels];

                      for(int j = 0; j < sizeof X_mem; ++j)
                        ((int8_t*)X_mem)[j] = (int8_t)pseudo_rand(&seed);


                      int8_t pad_val = (int8_t)pseudo_rand(&seed); //this should be unused in this case

                      Conv2dReorderedWeights rw = 
                        MatMulInt8::reorder_kernel_weights( (int8_t* )raw_weights, shape, 8, pad_val) ;

                      MatMulDirectFn::Params p(X, K, input_ch_per_output, rw.weights.data());
                      MatMulDirectFn mmd(&p);
                      
                      int ocg_count = (output_channels + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;
                      
                      for (int ocg = 0; ocg < ocg_count; ++ocg){

                        vpu_ring_buffer_t A;
                        // printf("start %p size:%d\n",  (int8_t*)X_mem, sizeof X_mem);
                        mmd.aggregate_fn(&A, (int8_t*)X_mem, ocg);

                        int chs_in_group = std::min(output_channels - vpu_ring_buffer_length *ocg , vpu_ring_buffer_length);

                        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan){

                          int actual_output_channel = output_chan + ocg * vpu_ring_buffer_length;

                          int expected_sum = 0;

                          for(int h = 0; h < k_height; ++h){
                            for(int w = 0 ; w < k_width; ++w){
                              for(int c = 0 ; c < input_ch_per_output; ++c){
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

    for (int x_channels = 1; x_channels <= 6; ++x_channels){
      for (int k_height = 1; k_height <= 6; ++k_height){
        for (int k_width = 1; k_width <= 6; ++k_width){
          for (int y_channels = 1; y_channels <= 6; ++y_channels){

            int8_t raw_weights[y_channels][k_height][k_width][x_channels];

            std::array<int, 4> shape = {y_channels, k_height, k_width, x_channels};
            int bits_per_element = 8;

            memset(raw_weights, 0, sizeof raw_weights); 

            Conv2dReorderedWeights rw = 
              MatMulInt8::reorder_kernel_weights((int8_t *)raw_weights, shape, bits_per_element, 0);

          }
        }
      }
    }
  }

  std::pair<double, double> pick_mul_and_bias(double output_min, double output_max, double accu_min, double accu_max, int * seed){
    double output_overscale = 1.1 + 0.2*(double)pseudo_rand(seed)/(double)INT32_MAX;
    double output_range = (output_max - output_min)*output_overscale;
    double mul = output_range / (double)(accu_max - (double)accu_min);
    double bias = output_min * output_overscale - accu_min * mul;
    return std::make_pair(mul, bias);
  }

  void pick_activation_params(
    std::vector<double>& multiplier, 
    std::vector<double>& bias, 
    std::vector<int32_t> &accu_min, 
    std::vector<int32_t> &accu_max, 
    int * seed)
  {

    double output_min = (double)INT8_MIN; 
    double output_max = (double)INT8_MAX; 

    std::tie(multiplier[0], bias[0]) = pick_mul_and_bias(output_min, output_max, accu_min[0], accu_max[0], seed);

    double max_abs_mul = std::abs(multiplier[0]);
    double min_abs_mul = max_abs_mul;

    for (int ch = 1; ch < multiplier.size(); ch++){

      std::tie(multiplier[ch], bias[ch]) = pick_mul_and_bias(output_min, output_max, accu_min[ch], accu_max[ch], seed);
      
      while(std::max(max_abs_mul, std::abs(multiplier[ch])) / std::min(min_abs_mul, std::abs(multiplier[ch])) > 63){
        std::tie(multiplier[ch], bias[ch]) = pick_mul_and_bias(output_min, output_max, accu_min[ch], accu_max[ch], seed);
      }
      max_abs_mul = std::max(max_abs_mul, std::abs(multiplier[ch]));
      min_abs_mul = std::min(min_abs_mul, std::abs(multiplier[ch]));
    }
  }

  void pick_accu_range(std::vector<int32_t> &accu_min, std::vector<int32_t> &accu_max, int * seed){
    
    //It's reasonable to assume all accumulators are approximatly of the same order
    int scale =  pseudo_rand(seed)%24;
    if(scale < 0)
      scale = -scale;
    scale += 1;

    int32_t a = 0;
    while (!a) a = pseudo_rand(seed)>>scale;

    for (int ch = 0; ch < accu_min.size(); ch++){

      int32_t b = ((pseudo_rand(seed)%a)>>(scale+2)) - a;

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

 class Test_OT_int8: public ::testing::Test {};

  TEST_F(Test_OT_int8, BasicTest) {
    
    const int vpu_ring_buffer_length = VPU_INT16_EPV;

    int seed = 69;

    for(int output_ch_count = 2; output_ch_count <= 64; ++output_ch_count){
      
      for (int itt=0;itt<1<<8;itt++){

        std::vector<double> f_biases(output_ch_count, 0);
        std::vector<double> f_multipliers(output_ch_count, 0);
        std::vector<int32_t> accu_min(output_ch_count, 0);
        std::vector<int32_t> accu_max(output_ch_count, 0);

        pick_accu_range(accu_min, accu_max, &seed);

        pick_activation_params(f_multipliers, f_biases, accu_max, accu_min, &seed);

        int t = ((output_ch_count + vpu_ring_buffer_length-1)/vpu_ring_buffer_length)*vpu_ring_buffer_length;

        int16_t accu_modifier[t]; //this comes from the aggregate fn
        memset(accu_modifier, 0, sizeof accu_modifier); 

        QuantisationParams qp = OTBinary_int8::quantise_activation(f_multipliers, f_biases, accu_min, accu_max);

        OTBinary_int8::Params p((int32_t)output_ch_count, &qp.otv, qp.biases.data(), 
          qp.multipliers.data(), (int16_t*)accu_modifier);
        OTBinary_int8 ot(&p);

        int8_t Y[output_ch_count];
        memset(Y, 0, sizeof Y); 

        int ocg_count = (output_ch_count + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;

        int8_t *y = (int8_t*)Y;

        for (int ocg = 0; ocg < ocg_count; ++ocg){

          int chs_in_group = std::min(output_ch_count - vpu_ring_buffer_length * ocg , vpu_ring_buffer_length);

          vpu_ring_buffer_t A;

          int8_t * next_y;
          for (int t=0;t<1<<8;t++){

            memset(&A, 0, sizeof A); 

            int32_t accu_values[chs_in_group];
            //fill A with random value between accu_max and accu_min
            for (int output_chan = 0; output_chan < chs_in_group; ++output_chan){

              int64_t range = (int64_t)accu_max[output_chan] - (int64_t)accu_min[output_chan];
              int32_t v = (int64_t)accu_min[output_chan] + ((unsigned)pseudo_rand(&seed))%range;
              
              accu_values[output_chan] = v;
              A.vR[output_chan] = ((int16_t *)&v)[1];
              A.vD[output_chan] = ((int16_t *)&v)[0];
            }

            next_y = ot.output_transform_fn(y, &A, ocg);

            for (int output_chan = 0; output_chan < chs_in_group; ++output_chan){

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



class MockMemCpyFn : public MemCpyFn {
  public:
    struct memcopy_fn_call {
      int8_t *T, *X;
      int32_t h, w, c;
    };

    struct {
      std::vector<memcopy_fn_call> memcopy_fn;
      int get_scratch_bytes;
      int get_overread_bytes;
    } calls;
public:
  MockMemCpyFn() : calls{ std::vector<memcopy_fn_call>(0), 0, 0 } { }
  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w, int32_t c) 
  { this->calls.memcopy_fn.push_back( memcopy_fn_call{T, X, h, w, c} ); return T; }
  size_t get_scratch_bytes(){ this->calls.get_scratch_bytes++; return 0; }
  size_t get_overread_bytes(){ this->calls.get_overread_bytes++; return 0; }
};

class MockAggregateFn : public AggregateFn {
public:
  struct aggregate_fn_call {
    vpu_ring_buffer_t * A;
    int8_t * T;
    int32_t output_channel_group;
  };

  struct {
    std::vector<aggregate_fn_call> aggregate_fn;
  } calls;

  MockAggregateFn() : calls{ std::vector<aggregate_fn_call>() } { }

  void aggregate_fn(vpu_ring_buffer_t * A , int8_t * T, int32_t output_channel_group)
  { this->calls.aggregate_fn.push_back( { A, T, output_channel_group } ); }
  
};

class MockOutputTransform : public OutputTransformFn {
  public:
    struct output_transform_fn_call {
      int8_t * Y;
      vpu_ring_buffer_t * A;
      int32_t output_channel_group;
    };

  private:
    int32_t output_slice_channel_count;

  public:
    struct {
      std::vector<output_transform_fn_call> output_transform_fn;
    } calls;
  
    MockOutputTransform(int32_t output_slice_channel_count) 
      : output_slice_channel_count(output_slice_channel_count),
        calls { std::vector<output_transform_fn_call>(0) } { }
        
    int8_t * output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)
    {
      this->calls.output_transform_fn.push_back( { Y, A, output_channel_group } );
      int output_count = std::min(output_slice_channel_count - output_channel_group * VPU_INT16_EPV, (int)VPU_INT16_EPV);
      for (int ch = 0; ch < output_count; ++ch)
        Y[ch] = 1;
      return Y + output_count;
    }
};


template <typename T>
class Filter2D_Test: public ::testing::Test {};

using Filter2D_Test_Types = ::testing::Types<Filter2D, Filter2D_DW>;
TYPED_TEST_SUITE(Filter2D_Test, Filter2D_Test_Types);

TYPED_TEST(Filter2D_Test, BasicTest) {
  
  const auto mult_memcopy = TypeParam::UsesPerGroupMemCopy;

  for (int y_height = 1; y_height <= 8; ++y_height){
    for (int y_width = 1; y_width <= 8; ++y_width){
      for (int y_channels = 1; y_channels <= 8; y_channels += 1){

        for (int r_height_start = 0; r_height_start < y_height; ++r_height_start){
          for (int r_height_end = r_height_start+1; r_height_end <= y_height; ++r_height_end){

            for (int r_width_start = 0; r_width_start < y_width; ++r_width_start){
              for (int r_width_end = r_width_start+1; r_width_end <= y_width; ++r_width_end){

                for (int r_channels_start = 0; r_channels_start < y_channels; ++r_channels_start){
                  for (int r_channels_end = r_channels_start+1; r_channels_end <= y_channels; ++r_channels_end){
                  
                    int const cog_size = VPU_INT8_ACC_PERIOD;

                    auto ir = nn::ImageRegion(r_height_start, r_width_start, r_channels_start,
                                              r_height_end - r_height_start,
                                              r_width_end - r_width_start,
                                              r_channels_end - r_channels_start);

                    auto ip = nn::ImageGeometry(y_height, y_width, y_channels);

                    const auto region_pixels = ir.PixelCount();
                    const auto cog_count = ir.ChannelOutputGroups(cog_size);

                    MockAggregateFn agg_fn;
                    MockMemCpyFn mem_fn;
                    MockOutputTransform ot_fn(r_channels_end - r_channels_start);

                    auto akp = typename AbstractKernel<TypeParam>::Params(ip, ir, cog_size);

                    TypeParam f(&akp, &mem_fn, &agg_fn, &ot_fn);

                    int8_t Y[y_height][y_width][y_channels];

                    std::memset(Y, 0, sizeof(Y));

                    f.execute((int8_t*)Y, nullptr);

  
#define ITER_MSG  "Y: " << ip << " | Output Region: " << ir

                    const auto expected_memcopy_calls = 
                      region_pixels * (mult_memcopy? cog_count : 1); 

                    ASSERT_EQ(mem_fn.calls.get_overread_bytes, 0) << ITER_MSG;
                    ASSERT_EQ(mem_fn.calls.get_scratch_bytes, 0) << ITER_MSG;
                    ASSERT_EQ(mem_fn.calls.memcopy_fn.size(), expected_memcopy_calls) << ITER_MSG;
                    ASSERT_EQ(agg_fn.calls.aggregate_fn.size(), region_pixels * cog_count) << ITER_MSG;
                    ASSERT_EQ(ot_fn.calls.output_transform_fn.size(), region_pixels * cog_count) << ITER_MSG;

                    //iterate through whohe output tensor
                    for(int y_h = 0; y_h < y_height; y_h++){
                      for(int y_w = 0; y_w < y_width; y_w++){
                        for(int y_ch = 0; y_ch < y_channels; y_ch++){
                          //check if the region is in the output space

                          auto in_region = ir.Within(y_h, y_w, y_ch);

                          int expected = in_region? 1 : 0;
                          int actual = Y[y_h][y_w][y_ch];

                          ASSERT_EQ(expected, actual) << ITER_MSG << " | Output Coords" 
                                                                  << nn::ImageVect(y_h, y_w, y_ch);
                        }
                      }
                    }
#undef ITER_MSG
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















  //end
}

int main(int argc, char **argv) {


    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
