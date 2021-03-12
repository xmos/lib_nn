#include <cstdint>
#include <cstring>

#ifndef IMAGE_HPP_
#define IMAGE_HPP_

struct padding_t {
    int top;
    int bottom;
    int left;
    int right;
};

struct WindowGeometry {

    struct {
        int height;
        int width;
    } shape;

    struct {
        int horizontal;
        int vertical;
    } stride;

    struct {
        int horizontal;
        int vertical;
    } dilation;

    WindowGeometry(
        int height,
        int width,
        int stride_horizontal,
        int stride_vertical,
        int dilation_horizontal, 
        int dilation_vertical
        ) 
{
  shape.height = height;
  shape.width = width;
  stride.horizontal = stride_horizontal;
  stride.vertical = stride_vertical;
  dilation.horizontal=dilation_horizontal;
  dilation.vertical=dilation_vertical;
}
};


#define CONV2D_OUTPUT_LENGTH(input_length, filter_size, dilation, stride)     \
  (((input_length - (filter_size + (filter_size - 1) * (dilation - 1)) + 1) + \
    stride - 1) /                                                             \
   stride)


struct ImageParams {

    int height;
    int width;
    int channels;
    size_t bits_per_element;

    ImageParams(
        int h,
        int w,
        int c,
        int bits_per_element) 
            : height(h), 
            width(w), 
            channels(c), 
            bits_per_element(bits_per_element) {}

    // Constructor for deriving the result of kernel K on input X
    ImageParams(ImageParams &X, WindowGeometry &K, int k_channels)
    {
        height = CONV2D_OUTPUT_LENGTH(X.height, K.shape.height, K.dilation.vertical, K.stride.vertical);
        width = CONV2D_OUTPUT_LENGTH(X.width, K.shape.width, K.dilation.horizontal, K.stride.horizontal);
        channels = k_channels;
        bits_per_element = X.bits_per_element;
    }
    
    ImageParams(ImageParams &X, WindowGeometry &K, int k_channels, padding_t &padding)
    {
        height = CONV2D_OUTPUT_LENGTH(X.height + padding.top + padding.bottom, 
            K.shape.height, K.dilation.vertical, K.stride.vertical);
        width = CONV2D_OUTPUT_LENGTH(X.width + padding.left + padding.right, 
            K.shape.width, K.dilation.horizontal, K.stride.horizontal);
        channels = k_channels;
        bits_per_element = X.bits_per_element;
    }
    
    int32_t const offset(
        int row,
        int col,
        int channel) const 
    {
        return channel + col * this->pixelBytes() + row * this->rowBytes();
    }

    size_t const pixelBytes() const { return (this->channels * bits_per_element + 7)/8;     }
    size_t const rowBytes()   const { return this->pixelBytes() * this->width;    }
    size_t const imageBytes() const { return this->rowBytes() * this->height;     }

};


#endif //IMAGE_HPP_