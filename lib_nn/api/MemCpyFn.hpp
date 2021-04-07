#include <cstdint>
#include <cstring>

#include "Image.hpp"

class MemCpyFn {
  public:

    //h, w, c are coordinates in the output space
    virtual int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w, int32_t c=0) = 0;
    virtual size_t get_scratch_bytes() = 0;
    virtual size_t get_overread_bytes() = 0;
};

class DerefInputFn : public MemCpyFn {

  public:
  class Params {
    public:
    int32_t bytes_per_h_line; 
    int32_t bytes_per_pixel; 
    Params(ImageParams &X, WindowGeometry &K){
      bytes_per_h_line = X.rowBytes() * K.stride.vertical;
      bytes_per_pixel = X.pixelBytes() * K.stride.horizontal; 
    }
  };

  Params * params;

  public:
  DerefInputFn(Params * params):params(params){};
  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w, int32_t c);
  size_t get_scratch_bytes();
  size_t get_overread_bytes();
};



class ImToColPadded : public MemCpyFn{
  public:
  class Params {

    public:
      int32_t kernel_height;
      int32_t kernel_width;

      int32_t vertical_stride;
      int32_t horizontal_stride;
      int32_t vertical_dilation;
      int32_t horizontal_dilation;

      padding_t padding;

      int32_t input_v_length;
      int32_t input_h_length;

      int32_t padding_val;

      int32_t bytes_per_h_line; 
      int32_t bytes_per_pixel;

      int32_t horizontal_mem_stride;

      int32_t bytes_per_copy_per_channel;

    public:
      Params(ImageParams &X, WindowGeometry &K, padding_t &padding, int input_ch_per_output, int8_t pad_val);
  };

  Params * params;

  public:
  ImToColPadded(Params * p):params(p){}
  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w, int32_t c);
  size_t get_scratch_bytes();
  size_t get_overread_bytes();
};

class ImToColValid : public MemCpyFn{
  public:
  struct Params {

    int32_t bytes_per_h_line; 
    int32_t bytes_per_pixel; 

    int32_t input_height; //in pixels
    int32_t input_width; //in pixels

    //This is the amount to copy in vpu words (round up)
    int32_t input_channel_groups; 

    int32_t T_rewind; 

    // The bytes to inc an X pointer by to move by one horizontal stride
    int32_t horizontal_mem_stride;

    // The bytes to inc an X pointer by to move by one vertical stride 
    // and horizontally bacwards by the kernel width.
    // i.e. from X[h][w + kernel_width - 1] to X[h+1][w]. 
    int32_t vertical_mem_stride;

    Params(ImageParams &X, WindowGeometry &K, int input_ch_per_output);
  };

  Params * params;

  public:

  //input_ch_per_output lets the kernel know how many input channels to copy to scratch
  ImToColValid(Params * params):params(params){};
  
  size_t get_scratch_bytes();
  size_t get_overread_bytes();

  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w, int32_t c);
  
};