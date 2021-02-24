#pragma once


namespace nn {
namespace filt2d {
namespace geom {


class WindowGeometry {

  public:

    struct {
      unsigned const height;
      unsigned const width;
    } shape;

    struct {
      int const row;
      int const col;
    } start;

    struct {
      int const row;    
      int const col;    
      int const channel;
    } stride;

    struct {
      int const row;
      int const col;
    } dilation;


  public:

    WindowGeometry(
      unsigned const height,
      unsigned const width,
      int const start_row = 0,
      int const start_col = 0,
      int const stride_rows = 1,
      int const stride_cols = 1,
      int const stride_chans = 0,
      int const dil_rows = 1,
      int const dil_cols = 1)
        : shape{height, width}, 
          start{start_row, start_col}, 
          stride{stride_rows, stride_cols, stride_chans}, 
          dilation{dil_rows, dil_cols} {}

};


}}}