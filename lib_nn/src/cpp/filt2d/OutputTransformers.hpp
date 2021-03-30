#pragma once

// #include "util.h"
// #include "Filter2d_util.hpp"
#include "geom/WindowGeometry.hpp"

namespace nn {

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



  EXTERN_C typedef struct {
      uint16_t shift1[ 16 ];
      int16_t  scale[ 16 ];
      int16_t  offset_scale[ 16 ];
      int16_t  offset[ 16 ];
      uint16_t shift2[ 16 ];
  } nn_acc32_to_int8_params_t;


  EXTERN_C void conv2d_output_transform_symmetric_int8(
        int8_t * output,
        const vpu_split_acc32_t* accumulator,
        const nn_acc32_to_int8_params_t* params,
        unsigned const channels_out);

  EXTERN_C void conv2d_output_transform_asymmetric_int8(
        int8_t * output,
        const vpu_split_acc32_t* accumulator,
        const nn_acc32_to_int8_params_t* params,
        unsigned const channels_out);




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



}
