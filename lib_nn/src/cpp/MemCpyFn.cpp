#include <cmath>
#include <cassert>
#include <climits>

#include "MemCpyFn.hpp"

#include "vpu_sim.h"

using namespace nn;

DerefInputFn::Params::Params(const int32_t bytes_per_h_line,
                             const int32_t bytes_per_pixel)
    : bytes_per_h_line(bytes_per_h_line), bytes_per_pixel(bytes_per_pixel)
{

}

DerefInputFn::Params::Params(const ImageGeometry& input, const WindowGeometry& window)
    : bytes_per_h_line(input.rowBytes() * window.stride.row), 
      bytes_per_pixel(input.pixelBytes() * window.stride.col)
{

}

DerefInputFn::Params::Params(const Filter2dGeometry& filter)
    : Params(filter.input, filter.window)
{

}

DerefInputFn::Params::Params(std::istream& stream)
{
#define READ_MEMBER(MEMBER)   stream.read(reinterpret_cast<char*>(&this->MEMBER), sizeof(this->MEMBER) )

  READ_MEMBER(bytes_per_h_line);
  READ_MEMBER(bytes_per_pixel);

#undef READ_MEMBER
}


void DerefInputFn::Params::Serialize(std::ostream& stream) const
{
#define WRITE_MEMBER(MEMBER)    stream.write(reinterpret_cast<const char*>(&this->MEMBER), sizeof(this->MEMBER) )

  WRITE_MEMBER(bytes_per_h_line);
  WRITE_MEMBER(bytes_per_pixel);

#undef WRITE_MEMBER
}

size_t DerefInputFn::get_scratch_bytes(){ return 0;}
size_t DerefInputFn::get_overread_bytes(){ return 0;}

int8_t * DerefInputFn::memcopy_fn(int8_t * T, 
                                  int8_t * X, 
                                  int32_t output_v_coord, 
                                  int32_t output_h_coord, 
                                  int32_t output_c_coord){

  return X + (int)(output_v_coord * params->bytes_per_h_line + 
    output_h_coord * params->bytes_per_pixel + output_c_coord);
}

size_t ImToColPadded::get_scratch_bytes(){
  return params->kernel_height * params->kernel_width * params->bytes_per_copy_per_channel + XS3_VPU_VREG_WIDTH_BYTES;
}

size_t ImToColPadded::get_overread_bytes(){
  return XS3_VPU_VREG_WIDTH_BYTES; //TODO this will be defined by the implementation of memcpy
}

ImToColPadded::Params::Params(const ImageGeometry &X, 
                              const WindowGeometry &K, 
                              const padding_t &padding, 
                              const int input_ch_per_output, 
                              const int8_t pad_val){

  kernel_height = K.shape.height;
  kernel_width = K.shape.width;

  vertical_stride = K.stride.row;
  horizontal_stride = K.stride.col;
  vertical_dilation = K.dilation.row;
  horizontal_dilation = K.dilation.col;

  this->padding = padding;

  input_v_length = X.height;
  input_h_length = X.width;

  padding_val = pad_val;

  bytes_per_pixel = X.pixelBytes();
  bytes_per_h_line = X.rowBytes();

  horizontal_mem_stride = X.rowBytes();

  //TODO 
  // bytes_per_copy_per_channel = (input_ch_per_output *  X.bits_per_element) / CHAR_BIT; 
  bytes_per_copy_per_channel = (input_ch_per_output *  CHAR_BIT) / CHAR_BIT; 
  
}

ImToColPadded::Params::Params(std::istream& stream)
{
#define READ_MEMBER(MEMBER)   stream.read(reinterpret_cast<char*>(&this->MEMBER), sizeof(this->MEMBER) )

  READ_MEMBER(kernel_height);
  READ_MEMBER(kernel_width);
  READ_MEMBER(vertical_stride);
  READ_MEMBER(horizontal_stride);
  READ_MEMBER(vertical_dilation);
  READ_MEMBER(horizontal_dilation);
  READ_MEMBER(padding);
  READ_MEMBER(input_v_length);
  READ_MEMBER(input_h_length);
  READ_MEMBER(padding_val);
  READ_MEMBER(bytes_per_h_line);
  READ_MEMBER(bytes_per_pixel);
  READ_MEMBER(horizontal_mem_stride);
  READ_MEMBER(bytes_per_copy_per_channel);

#undef READ_MEMBER
}

void ImToColPadded::Params::Serialize(std::ostream& stream) const
{
#define WRITE_MEMBER(MEMBER)    stream.write(reinterpret_cast<const char*>(&this->MEMBER), sizeof(this->MEMBER) )

  WRITE_MEMBER(kernel_height);
  WRITE_MEMBER(kernel_width);
  WRITE_MEMBER(vertical_stride);
  WRITE_MEMBER(horizontal_stride);
  WRITE_MEMBER(vertical_dilation);
  WRITE_MEMBER(horizontal_dilation);
  WRITE_MEMBER(padding);
  WRITE_MEMBER(input_v_length);
  WRITE_MEMBER(input_h_length);
  WRITE_MEMBER(padding_val);
  WRITE_MEMBER(bytes_per_h_line);
  WRITE_MEMBER(bytes_per_pixel);
  WRITE_MEMBER(horizontal_mem_stride);
  WRITE_MEMBER(bytes_per_copy_per_channel);

#undef WRITE_MEMBER
}

int8_t * ImToColPadded::memcopy_fn(int8_t * T, int8_t * X, 
int32_t output_v_coord, int32_t output_h_coord, int32_t output_c_coord){

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;
  int8_t * T_in = T;

  for(int32_t k_height = 0; k_height < params->kernel_height; k_height++){

    int32_t input_v_coord = (output_v_coord * params->vertical_stride + k_height * params->vertical_dilation);

    int p = input_v_coord < params->padding.top;
    p |= input_v_coord >= params->padding.top + params->input_v_length;

    int32_t input_h_coord = (output_h_coord * params->horizontal_stride );
    int8_t * X_cur_p = X + (int)((input_v_coord - params->padding.top) * params->bytes_per_h_line + (input_h_coord - params->padding.left) * params->bytes_per_pixel + output_c_coord);

    for(int32_t k_width = 0; k_width < params->kernel_width; k_width++){
      
      // int32_t input_h_coord = (output_h_coord * horizontal_stride + k_width * horizontal_dilation);
      int q = p;
      q |= input_h_coord < params->padding.left;
      q |= input_h_coord >= params->padding.left + params->input_h_length;

      //it might be nice to do a memcopy of the padding rather than a memset(requires more memory though)
      if(q){
        memset(T, params->padding_val, params->bytes_per_copy_per_channel);
      } else {
        // int8_t * X_cur_p = X + (int)((input_v_coord - padding.top) * bytes_per_h_line + (input_h_coord - padding.left) * bytes_per_pixel + output_c_coord);
        memcpy(T, X_cur_p, params->bytes_per_copy_per_channel);
      }

      T += params->bytes_per_copy_per_channel;

      //Advance the X_cur_p to the start of the next horizontal pixel
      // probably w_stride = nput_bytes_per_pixel * horizontal_stride
      X_cur_p += params->bytes_per_pixel * params->horizontal_dilation;
      input_h_coord += params->horizontal_dilation;

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
ImToColValid::Params::Params(ImageGeometry &X, WindowGeometry &K, int input_ch_per_output){

  //TODO
  // int bytes_per_copy_per_channel = (input_ch_per_output *  X.bits_per_element) / CHAR_BIT; 
  int bytes_per_copy_per_channel = (input_ch_per_output * CHAR_BIT) / CHAR_BIT; 

  bytes_per_pixel = X.pixelBytes();
  bytes_per_h_line = X.rowBytes(); 

  assert (X.rowBytes() == X.width * bytes_per_pixel);

  //This is the amount to copy in vpu words (round up)
  input_channel_groups = (bytes_per_copy_per_channel + XS3_VPU_VREG_WIDTH_BYTES-1)/XS3_VPU_VREG_WIDTH_BYTES;

  int bytes_actually_copied = input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES;
  T_rewind = bytes_actually_copied - bytes_per_copy_per_channel;

  input_height = K.shape.height;
  input_width  = K.shape.width;

  horizontal_mem_stride = bytes_per_pixel * K.dilation.col - bytes_actually_copied;
  vertical_mem_stride = bytes_per_h_line * K.dilation.row - input_width * bytes_per_pixel * K.dilation.col;

  //TODO rename these to account for the multiplication of strides
  bytes_per_h_line *= K.stride.row;
  bytes_per_pixel *= K.stride.col;

}

size_t ImToColValid::get_scratch_bytes(){
  return params->input_height * params->input_width * (params->input_channel_groups * XS3_VPU_VREG_WIDTH_BYTES - params->T_rewind) + XS3_VPU_VREG_WIDTH_BYTES;
}

size_t ImToColValid::get_overread_bytes(){
  return params->T_rewind;
}

int8_t * ImToColValid::memcopy_fn(int8_t * T, int8_t * X, 
  int32_t output_v_coord, int32_t output_h_coord, int32_t output_c_coord){

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  int8_t * X_cur_p = X + (int)(output_v_coord * params->bytes_per_h_line + output_h_coord * params->bytes_per_pixel + output_c_coord);
  
  int8_t * T_in = T;

  for(int32_t i_height = 0; i_height < params->input_height; i_height++){
    for(int32_t i_width = 0; i_width < params->input_width; i_width++){
      
      //This loop copies a whole pixel
      for(int32_t i_ch_group=0; i_ch_group < params->input_channel_groups; i_ch_group++){
        VLDD(vpu, X_cur_p);      
        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        VSTD(vpu, T);
        T += XS3_VPU_VREG_WIDTH_BYTES;
      }

      T -= params->T_rewind; 

      //Advance the X_cur_p to the start of the next horizontal pixel
      X_cur_p += params->horizontal_mem_stride;
    }

    //Advance the X_cur_p to the start of the next vertical pixel
    X_cur_p += params->vertical_mem_stride;
  }

  //Write padding to the tail, zeros is fastest
  VCLRDR(vpu);
  VSTD(vpu, T);

  return T_in;
}

