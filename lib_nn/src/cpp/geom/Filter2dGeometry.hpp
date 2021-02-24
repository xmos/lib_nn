#pragma once

#include "../Filter2d_util.hpp"
#include "ImageGeometry.hpp"
#include "WindowGeometry.hpp"

namespace nn {
namespace filt2d {
namespace geom {


template <typename T_input, typename T_output>
class Filter2dGeometry {

  public:

    ImageGeometry<T_input> const input;
    ImageGeometry<T_output> const output;
    WindowGeometry const window;


  public:

    Filter2dGeometry(
      ImageGeometry<T_input> const input_geom,
      ImageGeometry<T_output> const output_geom,
      WindowGeometry const window_geom)
        : input(input_geom), output(output_geom), window(window_geom) {}


    InputCoordTransform GetInputCoordTransform();

    PaddingTransform GetPaddingTransform();




};


}}}