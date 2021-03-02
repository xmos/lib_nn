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

template <typename T = int8_t>
class IMemCopyHandler {

  public:

    T const* copy_mem(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};


////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

template <typename T = int8_t>
class UniversalPatchMemCopyHandler : IMemCopyHandler<T>, IPaddingResolver { 

  protected:

    T * m_patch;
    PointerCovector m_input_covector;
    PointerCovector m_patch_covector;
    PaddingTransform m_padding_transform;

    unsigned m_K_h;
    unsigned m_K_w;
    unsigned m_K_d;
    T m_pad_value;


  public:

    UniversalPatchMemCopyHandler(
      unsigned K_h,
      unsigned K_w,
      unsigned K_d,
      T pad_value,
      T* scratch_mem,
      PointerCovector input_covector,
      PointerCovector patch_covector,
      PaddingTransform padding_transform) 
        : m_patch(scratch_mem), 
          m_input_covector(input_covector),
          m_patch_covector(patch_covector),
          m_padding_transform(padding_transform),
          m_K_h(K_h), m_K_w(K_w), m_K_d(K_d), m_pad_value(pad_value) {}

    PaddingTransform const& getPaddingTransform() const override {  return m_padding_transform;  }

    T const* copy_mem(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};


// Doesn't handle padding, depthwise stuff or dilation != 1
template <typename T = int8_t>
class ValidDeepMemCopyHandler : IMemCopyHandler<T> { 

  protected:

    T * m_patch;
    PointerCovector m_input_covector;
    const unsigned m_window_rows;
    const unsigned m_window_row_bytes;
    const unsigned m_img_row_bytes;


  public:

    template <typename T_out = int8_t>
    ValidDeepMemCopyHandler(T* patch, geom::Filter2dGeometry<T,T_out> const filter)
      : m_patch(patch), 
        m_input_covector(filter.input.getPointerCovector()), 
        m_window_rows(filter.window.shape.height),
        m_window_row_bytes(filter.window.rowBytes()),
        m_img_row_bytes(filter.input.rowBytes())
    {
      assert(filter.window.stride.channel == 0);
      assert(filter.window.dilation.row == 1);
      assert(filter.window.dilation.col == 1);
    }


    ValidDeepMemCopyHandler(
      unsigned window_rows,
      unsigned window_row_bytes,
      unsigned img_row_bytes,
      T* scratch_mem,
      PointerCovector input_covector) 
        : m_patch(scratch_mem), 
          m_input_covector(input_covector),
          m_window_rows(window_rows), 
          m_window_row_bytes(window_row_bytes),
          m_img_row_bytes(img_row_bytes) {}


    T const* copy_mem(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count);

};



}}