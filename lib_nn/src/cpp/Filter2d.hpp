#pragma once

#include "util.h"
#include "Filter2d_util.hpp"
#include "geom/Filter2dGeometry.hpp"

#include <type_traits>


namespace nn {
namespace filt2d {

template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans,
          class T_memcpy, class T_agg, class T_ot>
class Filter2d {

  static_assert(std::is_base_of<T_memcpy,IMemCopyHandler<T_elm_in>>::value, "T_memcpy must derive from IMemCopyHandler<T_elm_in>");
  static_assert(std::is_base_of<T_agg,IAggregationHandler<T_elm_in,T_acc>>::value, "T_agg must derive from IAggregationHandler<T_acc>>");
  static_assert(std::is_base_of<T_ot,IOutputTransformHandler<T_acc,T_elm_out>>::value, "T_ot must derive from IOutputTransformHandler<T_acc,T_elm_out>>");

  protected: 

    T_memcpy m_mem_copy;
    T_agg m_aggregator;
    T_ot m_output_transform;

    PointerCovector const m_output_covector;
    PointerCovector const m_input_covector;

    T_elm_in const* m_input_image;
    T_elm_out * m_output_image;

  protected:
    
    void computePixelSlice(
      int const out_row,
      int const out_col,
      int const out_chan_start,
      unsigned const out_chan_count) const;

    T_elm_in  const* getInputPointer(ImageVect const& output_coords) const;

    T_elm_out const* getOutputPointer(ImageVect const& output_coords) const;


  public:

    void bind(
        T_elm_in const* input_img,
        T_elm_out * output_img )
    {
      this->m_input_image = input_img;
      this->m_output_image = output_img;
    }

    void execute(ImageRegion const& job) const;

};



}}