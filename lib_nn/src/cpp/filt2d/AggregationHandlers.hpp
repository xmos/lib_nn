#pragma once

// #include "util.h"
#include "geom/WindowGeometry.hpp"

namespace nn {


  ////////////////////////////////////////////////////////
  /////
  ////////////////////////////////////////////////////////

  template <typename T_elm_in, 
            typename T_acc>
  class IAggregationHandler {


    public:

      T_acc aggregate(
        T_elm_in const* input_img,
        ImageVect const& output_coords,
        unsigned const channels_out);

  };



  ////////////////////////////////////////////////////////
  /////
  ////////////////////////////////////////////////////////



  EXTERN_C typedef struct {  
    uint32_t patch_bytes;  
    mem_stride_t K_cout_stride;
  } conv2d_aggregate_deep_patch_int8_params_t; 

  EXTERN_C void conv2d_aggregate_deep_patch_int8(
    vpu_split_acc32_t* accumulators,
    const int8_t* patch,
    const int8_t* kernel,
    const conv2d_aggregate_deep_patch_int8_params_t* params,
    const channel_count_t out_chans);




  template <typename T_elm_in = int8_t, 
            typename T_coef = int8_t, 
            typename T_acc = vpu_split_acc32_t>
  class Conv2dDeepPatchAggregator : public IAggregationHandler<int8_t, int8_t> {

    public:

      static constexpr unsigned MAX_COG_CHANS = 16;

      struct Config {
        const T_acc* biases;
        const T_coef* kernel_tensor;
        const unsigned elm_count;// i.e. # of T_elm_in's in the window
        const unsigned kernel_block_bytes; // i.e. bytes to advance one cog in the kernel tensor
        const conv2d_aggregate_deep_patch_int8_params_t agg_params;


        Config(
          const T_acc * biases,
          const T_coef * kernel_tensor,
          const unsigned window_elms,
          const unsigned kernel_block_bytes)
            : biases(biases),
              kernel_tensor(kernel_tensor),
              elm_count(window_elms),
              kernel_block_bytes(kernel_block_bytes),
              agg_params{(uint32_t) (window_elms * sizeof(int8_t)), (mem_stride_t) kernel_block_bytes} {}
        
        Config(
            const T_acc * biases,
            const T_coef * kernel_tensor,
            WindowGeometry const conv_window)
          : Config(biases, kernel_tensor, 
                   conv_window.shape.imageElements(), 
                   conv_window.shape.imageBytes() ) {}
            
          
      };

    protected:

      const Config& config;

    public: 


      Conv2dDeepPatchAggregator(
          const Config& conf)
        : config(conf) {}

      T_acc aggregate(
          T_elm_in const* input_img,
          ImageVect const& output_coords,
          unsigned const channels_out);



  };


}
