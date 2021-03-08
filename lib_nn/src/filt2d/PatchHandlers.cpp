
#include "PatchHandlers.hpp"

#include <iostream>
#include <cstring>

using namespace nn::filt2d;


template <typename T>
T const* UniversalPatchHandler<T>::copy_patch(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count)
{
  padding_t padding = this->getPadding(output_coords, true);


  unsigned row_bytes, pix_bytes;

  unsigned X_row_elms;


  for(int row = 0; row < m_K_h; row++){

    bool row_in_pad = (row < padding.top || row >= (m_K_h - padding.bottom));

    for(int col = 0; col < m_K_w; col++){
      
      bool col_in_pad = (col < padding.left || col >= (m_K_w - padding.right));
      bool pix_in_pad = row_in_pad || col_in_pad;

      if(pix_in_pad){
        for(int chan = 0; chan < m_K_d; chan++){

        }
      } else {
        for(int chan = 0; chan < m_K_d; chan++){

        }
      }
      


    }

  }

  return this->m_patch;
}



template int8_t const* UniversalPatchHandler<int8_t>::copy_patch(
      ImageVect const&, int8_t const*, unsigned const);



template <typename T>
T const* ValidDeepPatchHandler<T>::copy_patch(
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

template int8_t const* ValidDeepPatchHandler<int8_t>::copy_patch(
      ImageVect const&, int8_t const*, unsigned const);