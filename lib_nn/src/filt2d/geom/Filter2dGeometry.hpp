#pragma once

#include "../Filter2d_util.hpp"
#include "ImageGeometry.hpp"
#include "WindowGeometry.hpp"

namespace nn {
namespace filt2d {
namespace geom {


template <typename T_input = int8_t, typename T_output = int8_t>
class Filter2dGeometry {

  public:

    ImageGeometry<T_input> const input;
    ImageGeometry<T_output> const output;
    WindowGeometry<T_input> const window;


  public:

    Filter2dGeometry(
      ImageGeometry<T_input> const input_geom,
      ImageGeometry<T_output> const output_geom,
      WindowGeometry<T_input> const window_geom)
        : input(input_geom), output(output_geom), window(window_geom) {}

    InputCoordTransform GetInputCoordTransform();

    PaddingTransform GetPaddingTransform() const;

    const ImageRegion GetFullJob() const { return ImageRegion(0,0,0, output.height, output.width, output.channels); }



};





}}}