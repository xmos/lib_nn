#pragma once

#include "util.h"
#include "Filter2d_util.hpp"
#include "geom/WindowGeometry.hpp"

namespace nn {
namespace filt2d {




////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////


template <typename T_acc, typename T_out>
class IOutputTransformHandler {

  public:

    void transform(
      T_out * output,
      T_acc const& accumulator,
      ImageVect const& output_coords,
      unsigned const channels_out);

};



////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////



class Int8OutputTransformHandler : public IOutputTransformHandler<vpu_split_acc32_t,int8_t> {

  public:

    static constexpr unsigned MAX_COG_CHANS = 16;

    struct Config {
      const nn_acc32_to_int8_params_t* ot_params;
      const bool symmetric;

      Config(const nn_acc32_to_int8_params_t* ot_params, const bool symmetric)
        : ot_params(ot_params), symmetric(symmetric) {}
    };

  protected:

    const Config config;

  public: 

    Int8OutputTransformHandler(
      const Config& config)
        : config(config) {}

    void transform(
      int8_t * output,
      vpu_split_acc32_t const& accumulator,
      ImageVect const& output_coords,
      unsigned const channels_out);

};



}}
