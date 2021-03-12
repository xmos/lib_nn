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
      T const* input_img, // base address
      unsigned const out_chan_count);

};


////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

template <typename T = int8_t>
class UniversalPatchHandler : IPatchHandler<T> { 

  public:

    struct Config {

        const geom::ImageGeometry<T> input;
        const geom::WindowGeometry<T> window;

        const T padding_value;

        Config(const geom::ImageGeometry<T> input,
               const geom::WindowGeometry<T> window,
               const T padding_value)
          : input(input), window(window), padding_value(padding_value) {}
    };

  protected:

    const Config& config;
    T * patch_mem;

  public:

    UniversalPatchHandler(
      const Config& config,
      T * patch_mem)
        : config(config), patch_mem(patch_mem) {}

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