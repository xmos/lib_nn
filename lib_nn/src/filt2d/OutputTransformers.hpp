#pragma once

#include "util.h"
#include "Filter2d_util.hpp"
#include "geom/WindowGeometry.hpp"

namespace nn {
namespace filt2d {




////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////


template <typename T_acc = vpu_split_acc32_t, typename T_out = int8_t, unsigned N_cog_chans = 16>
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


class Int8OutputTransformHandler : public IOutputTransformHandler<vpu_split_acc32_t,int8_t,16> {

  protected:

    const nn_acc32_to_int8_params_t* m_ot_params;
    const bool m_symmetric;

  public: 

    Int8OutputTransformHandler(
      nn_acc32_to_int8_params_t const* ot_params,
      bool const symmetric = false)
        : m_ot_params(ot_params),
          m_symmetric(symmetric) {}

    void transform(
      int8_t * output,
      vpu_split_acc32_t const& accumulator,
      ImageVect const& output_coords,
      unsigned const channels_out);

};


}}
