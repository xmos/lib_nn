
#include "MemCopyHandlers.hpp"

#include <iostream>
#include <cstring>

using namespace nn::filt2d;


template <typename T>
T const* UniversalPatchMemCopyHandler<T>::copy_mem(
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



template int8_t const* UniversalPatchMemCopyHandler<int8_t>::copy_mem(
      ImageVect const&, int8_t const*, unsigned const);



template <typename T>
T const* ValidDeepMemCopyHandler<T>::copy_mem(
      ImageVect const& output_coords,
      T const* input_img,
      unsigned const out_chan_count)
{

  unsigned patch = (unsigned) this->m_patch;
  unsigned image = (unsigned) this->m_input_covector.resolve(input_img, output_coords);

  for(int row = this->m_window_rows; row > 0; row--){
    memcpy((void*)patch, (void*)image, this->m_window_row_bytes);
    patch += this->m_window_row_bytes;
    image += this->m_img_row_bytes;
  }

  return this->m_patch;
}

template int8_t const* ValidDeepMemCopyHandler<int8_t>::copy_mem(
      ImageVect const&, int8_t const*, unsigned const);