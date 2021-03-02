#pragma once

#include "util.h"
#include "Filter2d_util.hpp"
#include "geom/WindowGeometry.hpp"

namespace nn {
namespace filt2d {




////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

template <typename T_elm_in = int8_t, 
          typename T_acc = vpu_split_acc32_t, 
          unsigned N_cog_chans = 16>
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

template <typename T_elm_in = int8_t, 
          typename T_coef = int8_t, 
          typename T_acc = vpu_split_acc32_t, 
          unsigned N_cog_chans = 16>
class Conv2dDeepPatchAggregator : public IAggregationHandler<T_elm_in, T_acc, N_cog_chans> {

  protected:

    const T_acc* m_biases;
    const T_coef* m_kernel_tensor;
    const unsigned m_elm_count; // i.e. # of T_elm_in's in the window
    const unsigned m_kernel_block_bytes; // i.e. bytes to advance one cog in the kernel tensor
    const conv2d_aggregate_deep_patch_int8_params_t m_agg_params;

  public: 

    Conv2dDeepPatchAggregator(
      T_acc const* biases,
      T_coef const* kernel_tensor,
      unsigned window_elms,
      unsigned kernel_block_bytes)
        : m_biases(biases),
          m_kernel_tensor(kernel_tensor),
          m_elm_count(window_elms),
          m_kernel_block_bytes(kernel_block_bytes),
          m_agg_params{window_elms * sizeof(int8_t), (mem_stride_t) kernel_block_bytes} {}

    Conv2dDeepPatchAggregator(
      T_acc const* biases,
      T_coef const* kernel_tensor,
      geom::WindowGeometry<T_elm_in> const conv_window)
        : Conv2dDeepPatchAggregator(biases, kernel_tensor, 
                                    conv_window.windowElements(), 
                                    conv_window.windowBytes()) {}

    T_acc aggregate(
        T_elm_in const* input_img,
        ImageVect const& output_coords,
        unsigned const channels_out);



};


}}
