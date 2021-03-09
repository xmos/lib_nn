#include <cstdint>
#include <cstring>

#include "Image.hpp"

class MemCpyFn {
  public:
    virtual int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w) = 0;
    virtual size_t get_scratch_bytes() = 0;
    virtual size_t get_overread_bytes() = 0;
};

class NopValid : public MemCpyFn{

  int32_t bytes_per_h_line; 
  int32_t bytes_per_pixel; 

  public:
  NopValid(ImageParams &X);
  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w);
  size_t get_scratch_bytes();
  size_t get_overread_bytes();
};

class Im_to_col_padded : public MemCpyFn{
  int32_t kernel_height;
  int32_t kernel_width;

  int32_t vertical_stride;
  int32_t horizontal_stride;

  padding_t padding;

  int32_t input_v_length;
  int32_t input_h_length;

  int32_t padding_val;

  int32_t bytes_per_pixel;

  size_t horizontal_mem_stride;
  public:
  Im_to_col_padded(ImageParams &X, ImageParams &Y, WindowGeometry &K, padding_t &padding);
  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w);
  size_t get_scratch_bytes();
  size_t get_overread_bytes();
};

class Im_to_col_valid : public MemCpyFn{

  int32_t bytes_per_h_line; 
  int32_t bytes_per_pixel; 

  int32_t kernel_height; //in pixels
  int32_t kernel_width; //in pixels

  int32_t kernel_channel_groups;

  int32_t T_rewind; 

  // The bytes to inc an X pointer by to move by one horizontal stride
  int32_t horizontal_mem_stride;

  // The bytes to inc an X pointer by to move by one vertical stride 
  // and horizontally bacwards by the kernel width.
  // i.e. from X[h][w + kernel_width - 1] to X[h+1][w]. 
  int32_t vertical_mem_stride;

  public:
  Im_to_col_valid(ImageParams &X, ImageParams &Y, WindowGeometry &K);
  size_t get_scratch_bytes();
  size_t get_overread_bytes();

  int8_t * memcopy_fn(int8_t * T, int8_t * X, int32_t h, int32_t w);
};