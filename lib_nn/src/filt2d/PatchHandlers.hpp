#pragma once

#include "util.h"
#include "Filter2d_util.hpp"
#include "geom/Filter2dGeometry.hpp"

#include <cassert>


namespace nn {
namespace filt2d {

////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

template <typename T>
class IPatchHandler {

  public:

    T const* copy_patch(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};


////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

template <typename T = int8_t>
class UniversalPatchHandler : IPatchHandler<T>, IPaddingResolver { 

  public:

    struct Config {

        /*
          What do I need to make this work?
            - Given:
              - Input image base address
              - Patch base address
              - Padding computation
            - Address in input image at which to start copying data
            - Address in patch at which the data starts
            - How many rows of pixels to copy over
            - Stride between rows of the patch
            - Either: 
              - If supporting horizontal dilation > 1:
                - Number of pixels to copy for each row not fully in padding
                - stride between pixels to copy (e.g. channel count * dilation)
                - stride (in input image) between rows to copy (e.g. input rowBytes * vertical dilation) 
              - otherwise (no horizontal dilation support):
                - bytes to be copied per copied row
                - stride between rows of the input image (which can account for vertical dilation)




          input_address = input_address_covector.resolve(input_image_base, 
                                  slice_coords + (unsigned_pad.top, unsigned_pad.left, 0))  

          


        */

    };

  protected:

    T * m_patch;
    AddressCovector<T> m_input_covector;
    AddressCovector<T> m_patch_covector;
    PaddingTransform m_padding_transform;

    unsigned m_K_h;
    unsigned m_K_w;
    unsigned m_K_d;
    T m_pad_value;

  public:

    UniversalPatchHandler(
      unsigned K_h,
      unsigned K_w,
      unsigned K_d,
      T pad_value,
      T* scratch_mem,
      AddressCovector<T> input_covector,
      AddressCovector<T> patch_covector,
      PaddingTransform padding_transform) 
        : m_patch(scratch_mem), 
          m_input_covector(input_covector),
          m_patch_covector(patch_covector),
          m_padding_transform(padding_transform),
          m_K_h(K_h), m_K_w(K_w), m_K_d(K_d), m_pad_value(pad_value) {}

    PaddingTransform const& getPaddingTransform() const override {  return m_padding_transform;  }

    T const* copy_patch(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};



////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

// Doesn't handle padding, depthwise stuff or dilation != 1
template <typename T = int8_t>
class ValidDeepPatchHandler : IPatchHandler<T> {

  public:

    struct Config {
      AddressCovector<T> input_covector;
      unsigned window_rows;
      unsigned window_row_bytes;
      unsigned img_row_bytes;

      Config(
        const AddressCovector<T>& input_covector,
        const unsigned window_rows,
        const unsigned window_row_bytes,
        const unsigned img_row_bytes)
          : input_covector(input_covector),
            window_rows(window_rows),
            window_row_bytes(window_row_bytes),
            img_row_bytes(img_row_bytes) {}

      template<typename T_out = int8_t>
      Config(geom::Filter2dGeometry<T,T_out> const filter)
        : Config(filter.input.getAddressCovector(), 
                 filter.window.shape.height,
                 filter.window.rowBytes(),
                 filter.input.rowBytes()) {}
    };  

  protected:

    const Config& config;

    T * patch_mem;

  public:

    ValidDeepPatchHandler(
        const Config& config,
        T* scratch_mem)
      : config(config),
        patch_mem(scratch_mem) {}


    T const* copy_patch(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};


}}