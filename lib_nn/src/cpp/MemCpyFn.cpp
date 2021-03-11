#include <cmath>
#include <cassert>
#include "MemCpyFn.hpp"

extern "C" {
  #include "vpu_sim.h"
}
#include <iostream>
#include <stdio.h>

NopValid::NopValid(ImageParams &X){
  bytes_per_h_line = X.rowBytes(); 
  bytes_per_pixel = X.pixelBytes();
}

size_t NopValid::get_scratch_bytes(){
  return 0;
}

size_t NopValid::get_overread_bytes(){
  return 0;
}

int8_t * NopValid::memcopy_fn(int8_t * T, int8_t * X, int32_t output_v_coord, int32_t output_h_coord){
  return X + output_v_coord * bytes_per_h_line + output_h_coord * bytes_per_pixel;
}

Im_to_col_padded::Im_to_col_padded(ImageParams &X, ImageParams &Y, WindowGeometry &K, padding_t &padding){

  kernel_height = K.shape.height;
  kernel_width = K.shape.width;

  vertical_stride = K.stride.vertical;
  horizontal_stride = K.stride.horizontal;

  this->padding = padding;

  input_v_length = X.height;
  input_h_length = X.width;

  padding_val = 0;

  bytes_per_pixel = X.pixelBytes();
  horizontal_mem_stride = X.rowBytes();
  
}

size_t Im_to_col_padded::get_scratch_bytes(){
  return kernel_height * kernel_width * bytes_per_pixel + XS3_VPU_VREG_WIDTH_BYTES;
}

size_t Im_to_col_padded::get_overread_bytes(){
  return XS3_VPU_VREG_WIDTH_BYTES; //TODO
}

int8_t * Im_to_col_padded::memcopy_fn(int8_t * T, int8_t * X, int32_t output_v_coord, int32_t output_h_coord){
  
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  for(int32_t k_height = 0; k_height < kernel_height; k_height++){

    int bytes_per_h_line = bytes_per_pixel * (4);//TODO

    int8_t * X_cur_p = X + (output_v_coord - padding.top) * bytes_per_h_line + 
      (output_h_coord - padding.left) * bytes_per_pixel;

    for(int32_t k_width = 0; k_width < kernel_width; k_width++){
      
      int32_t input_v_coord = output_v_coord * vertical_stride;
      int32_t input_h_coord = output_h_coord * horizontal_stride;

      int p = 0;
      p |= input_v_coord < padding.top;
      p |= input_v_coord + input_v_length > padding.bottom;
      p |= input_h_coord < padding.left;
      p |= input_h_coord + input_h_length > padding.right;

      //it might be nice to do a memcopy of the padding rather than a memset(requires more memory though)
      if(p){
        memset(T, padding_val, bytes_per_pixel);
      } else {
        //translate X to the actual input pointer
        memcpy(T, X_cur_p, bytes_per_pixel);
      }

      T += bytes_per_pixel;

      //Advance the X_cur_p to the start of the next horizontal pixel
      // probably w_stride = nput_bytes_per_pixel * horizontal_stride
      X_cur_p += horizontal_mem_stride;
    }
    //There is no vertical mem stride as X_cur_p is recalculated each step of the kernel height
  }

  //Write padding to the tail, zeros is fastest
  VCLRDR(vpu);
  VSTD(vpu, T);

  return T;//wrong t_in
}

/*
This constructor is used for testing
*/
Im_to_col_valid::Im_to_col_valid(ImageParams &X, WindowGeometry &K){
  
  bytes_per_pixel = X.pixelBytes();
  bytes_per_h_line = X.rowBytes(); 

  assert (X.rowBytes() == X.width * bytes_per_pixel);

  input_channel_groups = (bytes_per_pixel + XS3_VPU_VREG_WIDTH_BYTES-1)/XS3_VPU_VREG_WIDTH_BYTES;

  int bytes_actually_copied = input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES;
  T_rewind = bytes_actually_copied - bytes_per_pixel;

  input_height = K.shape.height;
  input_width  = K.shape.width;

  horizontal_mem_stride = bytes_per_pixel * K.dilation.horizontal - bytes_actually_copied;
  vertical_mem_stride = bytes_per_h_line * K.dilation.vertical - input_width * bytes_per_pixel * K.dilation.horizontal;

}

size_t Im_to_col_valid::get_scratch_bytes(){
  return input_height * input_width * (input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES - T_rewind) + XS3_VPU_VREG_WIDTH_BYTES;
}

size_t Im_to_col_valid::get_overread_bytes(){
  return input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES - bytes_per_pixel;
}

int8_t * Im_to_col_valid::memcopy_fn(int8_t * T, int8_t * X, int32_t output_v_coord, int32_t output_h_coord){

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  // std::cout << "init X_cur[0]:" << (int)X[0] << std::endl;

  //translate X to the actual input pointer, i.e. X_t = X[h][w]
  // std::cout << "move: " <<(output_v_coord * bytes_per_h_line + output_h_coord * bytes_per_pixel)<<std::endl;
  // printf("init X %p\n", X );
  int8_t * X_cur_p = X + (int)(output_v_coord * bytes_per_h_line + output_h_coord * bytes_per_pixel);
  
  // printf("X_cur %p\n", X_cur_p );
  // std::cout << "X_cur[0]:" << (int)X_cur_p[0] << std::endl;
  int8_t * T_in = T;

  for(int32_t i_height = 0; i_height < input_height; i_height++){
    for(int32_t i_width = 0; i_width < input_width; i_width++){
      
      //This loop copies a whole pixel
      for(int32_t i_ch_group=0; i_ch_group < input_channel_groups; i_ch_group++){
        VLDD(vpu, X_cur_p);      
        // std::cout << "XS3_VPU_VREG_WIDTH_BYTES:"<< XS3_VPU_VREG_WIDTH_BYTES << std::endl;
        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        VSTD(vpu, T);
        T += XS3_VPU_VREG_WIDTH_BYTES;
      }

      T -= T_rewind; 

      //Advance the X_cur_p to the start of the next horizontal pixel
      // std::cout << "horizontal_mem_stride:"<< horizontal_mem_stride << std::endl;
      X_cur_p += horizontal_mem_stride;
    }

    //Advance the X_cur_p to the start of the next vertical pixel
    // std::cout << "vertical_mem_stride:"<< vertical_mem_stride << std::endl;
    X_cur_p += vertical_mem_stride;
  }

  //Write padding to the tail, zeros is fastest
  VCLRDR(vpu);
  VSTD(vpu, T);

  return T_in;
}

