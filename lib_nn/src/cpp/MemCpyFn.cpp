#include <cmath>
#include <cassert>
#include "MemCpyFn.hpp"

extern "C" {
  #include "vpu_sim.h"
}
#include <climits>
#include <iostream>
#include <stdio.h>

DerefInputFn::DerefInputFn(ImageParams &X, WindowGeometry &K){
  bytes_per_h_line = X.rowBytes(); 
  bytes_per_pixel = X.pixelBytes();
  //TODO deal with strides
}

size_t DerefInputFn::get_scratch_bytes(){ return 0;}
size_t DerefInputFn::get_overread_bytes(){ return 0;}

int8_t * DerefInputFn::memcopy_fn(int8_t * T, int8_t * X, 
  int32_t output_v_coord, int32_t output_h_coord, int32_t output_c_coord){

  //TODO this needs stride
  //TODO make this into a base class function
  return X + output_v_coord * bytes_per_h_line + output_h_coord * bytes_per_pixel+ output_c_coord;
}

size_t Im_to_col_padded::get_scratch_bytes(){
  return kernel_height * kernel_width * bytes_per_copy_per_channel + XS3_VPU_VREG_WIDTH_BYTES;
}

size_t Im_to_col_padded::get_overread_bytes(){
  return XS3_VPU_VREG_WIDTH_BYTES; //TODO this will be defined by the implementation of memcpy
}

Im_to_col_padded::Im_to_col_padded(ImageParams &X, WindowGeometry &K, padding_t &padding, int input_ch_per_output, int8_t pad_val){


  kernel_height = K.shape.height;
  kernel_width = K.shape.width;

  vertical_stride = K.stride.vertical;
  horizontal_stride = K.stride.horizontal;
  vertical_dilation = K.dilation.vertical;
  horizontal_dilation = K.dilation.horizontal;

  this->padding = padding;

  input_v_length = X.height;
  input_h_length = X.width;

  padding_val = pad_val;

  bytes_per_pixel = X.pixelBytes();
  bytes_per_h_line = X.rowBytes();

  horizontal_mem_stride = X.rowBytes();

  bytes_per_copy_per_channel = (input_ch_per_output *  X.bits_per_element) / CHAR_BIT; 
  
}


int8_t * Im_to_col_padded::memcopy_fn(int8_t * T, int8_t * X, 
int32_t output_v_coord, int32_t output_h_coord, int32_t output_c_coord){

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;
  int8_t * T_in = T;

  for(int32_t k_height = 0; k_height < kernel_height; k_height++){

    for(int32_t k_width = 0; k_width < kernel_width; k_width++){
      
      int32_t input_v_coord = (output_v_coord * vertical_stride + k_height * vertical_dilation);
      int32_t input_h_coord = (output_h_coord * horizontal_stride+ k_width * horizontal_dilation);

      int p = input_v_coord < padding.top;
      p |= input_v_coord >= padding.top + input_v_length;
      p |= input_h_coord < padding.left;
      p |= input_h_coord >= padding.left + input_h_length;

      //it might be nice to do a memcopy of the padding rather than a memset(requires more memory though)
      if(p){
        // std::cout << "padding" <<std::endl;
        memset(T, padding_val, bytes_per_copy_per_channel);
      } else {
        int8_t * X_cur_p = X + (int)((input_v_coord - padding.top) * bytes_per_h_line + (input_h_coord - padding.left) * bytes_per_pixel + output_c_coord);

        // std::cout << " input_v_coord " << input_v_coord << " input_h_coord " << input_h_coord << std::endl;

        // for (int i=0;i<bytes_per_copy_per_channel;i++)
        //   std::cout << (int)X_cur_p[i] << std::endl;
        //translate X to the actual input pointer
        memcpy(T, X_cur_p, bytes_per_copy_per_channel);
      }

      T += bytes_per_copy_per_channel;

      //Advance the X_cur_p to the start of the next horizontal pixel
      // probably w_stride = nput_bytes_per_pixel * horizontal_stride
      // X_cur_p += horizontal_mem_stride;

    }
    //There is no vertical mem stride as X_cur_p is recalculated each step of the kernel height
  }

  //Write padding to the tail, zeros is fastest
  VCLRDR(vpu);
  VSTD(vpu, T);

  return T_in;//wrong t_in
}

/*
This constructor is used for testing
*/
Im_to_col_valid::Im_to_col_valid(ImageParams &X, WindowGeometry &K, int input_ch_per_output){

  int bytes_per_copy_per_channel = (input_ch_per_output *  X.bits_per_element) / CHAR_BIT; 

  bytes_per_pixel = X.pixelBytes();
  bytes_per_h_line = X.rowBytes(); 

  assert (X.rowBytes() == X.width * bytes_per_pixel);

  //This is the amount to copy in vpu words (round up)
  input_channel_groups = (bytes_per_copy_per_channel + XS3_VPU_VREG_WIDTH_BYTES-1)/XS3_VPU_VREG_WIDTH_BYTES;

  int bytes_actually_copied = input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES;
  T_rewind = bytes_actually_copied - bytes_per_copy_per_channel;

  input_height = K.shape.height;
  input_width  = K.shape.width;

  horizontal_mem_stride = bytes_per_pixel * K.dilation.horizontal - bytes_actually_copied;
  vertical_mem_stride = bytes_per_h_line * K.dilation.vertical - input_width * bytes_per_pixel * K.dilation.horizontal;

  //TODO rename these to account for the multiplication of strides
  bytes_per_h_line *=  K.stride.vertical;
  bytes_per_pixel *=  K.stride.horizontal;

}

size_t Im_to_col_valid::get_scratch_bytes(){
  return input_height * input_width * (input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES - T_rewind) + XS3_VPU_VREG_WIDTH_BYTES;
}

size_t Im_to_col_valid::get_overread_bytes(){
  return T_rewind;
}

int8_t * Im_to_col_valid::memcopy_fn(int8_t * T, int8_t * X, 
  int32_t output_v_coord, int32_t output_h_coord, int32_t output_c_coord){

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  int8_t * X_cur_p = X + (int)(output_v_coord * bytes_per_h_line + output_h_coord * bytes_per_pixel + output_c_coord);
  
  int8_t * T_in = T;

  for(int32_t i_height = 0; i_height < input_height; i_height++){
    for(int32_t i_width = 0; i_width < input_width; i_width++){
      
      //This loop copies a whole pixel
      for(int32_t i_ch_group=0; i_ch_group < input_channel_groups; i_ch_group++){
        VLDD(vpu, X_cur_p);      
        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        VSTD(vpu, T);
        T += XS3_VPU_VREG_WIDTH_BYTES;
      }

      T -= T_rewind; 

      //Advance the X_cur_p to the start of the next horizontal pixel
      X_cur_p += horizontal_mem_stride;
    }

    //Advance the X_cur_p to the start of the next vertical pixel
    X_cur_p += vertical_mem_stride;
  }

  //Write padding to the tail, zeros is fastest
  VCLRDR(vpu);
  VSTD(vpu, T);

  return T_in;
}

