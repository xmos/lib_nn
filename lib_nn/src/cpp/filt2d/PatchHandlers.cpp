
#include "PatchHandlers.hpp"

#include <iostream>
#include <cstring>

using namespace nn;


UniversalPatchHandler::T const* UniversalPatchHandler::copy_patch(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count)
{

  const auto input_cov = config.input.getAddressCovector<int8_t>();
  const auto patch_cov = config.window.shape.getAddressCovector<int8_t>();

  for(int row = 0; row < config.window.shape.height; row++){
    for(int col = 0; col < config.window.shape.width; col++){
      for(int chan = 0; chan < config.window.shape.depth; chan++){

        const int x_row = config.window.start.row 
                          + output_coords.row * config.window.stride.row 
                          + row * config.window.dilation.row;

        const int x_col = config.window.start.col
                          + output_coords.col * config.window.stride.col
                          + row * config.window.dilation.col;

        const int x_chan = output_coords.channel * config.window.stride.channel + chan;

        const bool in_padding =  ((x_row < 0) || (x_row >= config.input.height)) 
                              || ((x_col < 0) || (x_col >= config.input.width ));

        T* patch_add = patch_cov.resolve(this->patch_mem, row, col, chan);

        patch_add[0] = in_padding? config.padding_value : input_cov.resolve(input_img, x_row, x_col, x_chan)[0];


      }
    }
  }

  return this->patch_mem;
}



ValidDeepPatchHandler::T const* ValidDeepPatchHandler::copy_patch(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count)
{

  T* patch = this->patch_mem;
  T* image = this->config.input_covector.resolve(input_img, output_coords);

  for(int row = this->config.window_rows; row > 0; row--){
    memcpy(patch, image, this->config.window_row_bytes);
    patch = advancePointer(patch, this->config.window_row_bytes);
    image = advancePointer(image, this->config.img_row_bytes);
  }

  return this->patch_mem;
}