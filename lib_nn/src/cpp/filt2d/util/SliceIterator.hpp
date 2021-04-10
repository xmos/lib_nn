#pragma once

// #include "Filter2d_util.hpp"
#include "../geom/Filter2dGeometry.hpp"
#include "../PatchHandlers.hpp"
#include "../AggregationHandlers.hpp"
#include "../OutputTransformers.hpp"

#include <iterator>
#include <cstddef>

namespace nn {

struct SliceType {
  struct Channel {};
  struct Column {};
};

template <class T, typename SliceTypeTag>
struct SliceIterator
{

  protected:
    T m_job;

    const ImageVect m_start_vect;
    const ImageVect m_end_vect;

  public:
    SliceIterator(T job, 
                  const ImageRegion& region)
      : m_job(job), 
        m_start_vect(region.startVect()), 
        m_end_vect(region.endVect()) {}

    void execute() {
      this->exec(SliceTypeTag());
    }

  protected:

    void exec(SliceType::Channel)
    {
      ImageVect slice_coords(m_start_vect);
      const unsigned slice_chans = this->m_end_vect.channel - this->m_start_vect.channel;

      for(slice_coords.row = m_start_vect.row; slice_coords.row < m_end_vect.row; slice_coords.row++){
        for(slice_coords.col = m_start_vect.col; slice_coords.col < m_end_vect.col; slice_coords.col++){
          m_job.computeSlice(slice_coords, slice_chans);
        }
      }
    }

    void exec(SliceType::Column)
    {
      ImageVect slice_coords(m_start_vect);
      const unsigned slice_cols = this->m_end_vect.col - this->m_start_vect.col;

      for(slice_coords.channel = m_start_vect.channel; 
                    slice_coords.channel < m_end_vect.channel; slice_coords.channel++){
        for(slice_coords.row = m_start_vect.row; slice_coords.row < m_end_vect.row; slice_coords.row++){
          m_job.computeSlice(slice_coords, slice_cols);
        }
      }
    }


};


}