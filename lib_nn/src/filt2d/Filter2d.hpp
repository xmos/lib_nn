#pragma once

#include "util.h"
#include "Filter2d_util.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "MemCopyHandlers.hpp"
#include "AggregationHandlers.hpp"
#include "OutputTransformers.hpp"

#include <type_traits>


namespace nn {
namespace filt2d {

template <typename T_elm_in,  typename T_elm_out, typename T_acc, unsigned N_cog_chans,
          class T_memcpy, 
          class T_agg, 
          class T_ot>
class Filter2d {

  static_assert(std::is_base_of<IMemCopyHandler<T_elm_in>,T_memcpy>::value, 
                "T_memcpy must derive from IMemCopyHandler<T_elm_in>");
  static_assert(std::is_base_of<IAggregationHandler<T_elm_in,T_acc,N_cog_chans>,T_agg>::value, 
                "T_agg must derive from IAggregationHandler<T_elm_in,T_acc,N_cog_chans>>");
  static_assert(std::is_base_of<IOutputTransformHandler<T_acc,T_elm_out,N_cog_chans>,T_ot>::value, 
                "T_ot must derive from IOutputTransformHandler<T_acc,T_elm_out,N_cog_chans>>");


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
      unsigned const out_chan_count);

    T_elm_in  const* getInputPointer(ImageVect const& output_coords) const;

    T_elm_out* getOutputPointer(ImageVect const& output_coords) const;


  public:

    Filter2d(
      T_memcpy mem_copy_handler,
      T_agg aggregation_handler,
      T_ot output_transform_handler,
      PointerCovector output_pointer_covector,
      PointerCovector input_pointer_covector)
        : m_mem_copy(mem_copy_handler),
          m_aggregator(aggregation_handler),
          m_output_transform(output_transform_handler),
          m_output_covector(output_pointer_covector),
          m_input_covector(input_pointer_covector) {}

    Filter2d(
      T_memcpy mem_copy_handler,
      T_agg aggregation_handler,
      T_ot output_transform_handler,
      geom::Filter2dGeometry<T_elm_in,T_elm_out> const filt2d)
        : Filter2d(mem_copy_handler, aggregation_handler, output_transform_handler, 
                   filt2d.output.getPointerCovector(), filt2d.input.getPointerCovector()) {}


    void bind(
        T_elm_in const* input_img,
        T_elm_out * output_img )
    {
      this->m_input_image = input_img;
      this->m_output_image = output_img;
    }

    void execute(ImageRegion const& job);

};




template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans, class T_memcpy, class T_agg, class T_ot>
void Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::execute(
      ImageRegion const& job)
{ 
  // Default implementation iterates over the output rows and columns of job.
  //   - By iterating over channels within computePixelSlice() we only have to copy the patch once
  for(int row = job.start.row; row < (job.start.row + job.shape.height); row++)
    for(int col = job.start.col; col < (job.start.col + job.shape.width); col++)
      this->computePixelSlice(row, col, job.start.channel, job.shape.depth);
}



template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans, class T_memcpy, class T_agg, class T_ot>
T_elm_in  const* Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::getInputPointer(
    ImageVect const& output_coords) const
{ 
  return this->m_input_covector.resolve(this->m_input_image, output_coords);
}


template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans, class T_memcpy, class T_agg, class T_ot>
T_elm_out * Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::getOutputPointer(
    ImageVect const& output_coords) const
{
  return this->m_output_covector.resolve(this->m_output_image, output_coords);
}


template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans, class T_memcpy, class T_agg, class T_ot>
void Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::computePixelSlice(
      int const out_row,
      int const out_col,
      int const out_chan_start,
      unsigned const out_chan_count)
{
  // Upper bound of output channels to be processed by this call.
  int const last_chan = out_chan_start + out_chan_count;

  // Copy the intersection of the input image and the convolution window into the patch memory, filling in with padding 
  // as needed.
  auto const* input_src = this->m_mem_copy.copy_mem(
      ImageVect(out_row, out_col, out_chan_start), 
      this->m_input_image, out_chan_count);

  // Iterate over channels of the output pixel, handling N_cog_chans channels per iteration (except possibly the last)
  for(int out_chan = out_chan_start; out_chan < last_chan; out_chan += N_cog_chans){
    
    //Output channels for this iteration
    unsigned const iter_chans = (last_chan - out_chan >= N_cog_chans)? N_cog_chans : (last_chan - out_chan);

    //Output coordinates at which this pixel slice starts
    ImageVect const output_coords = ImageVect(out_row, out_col, out_chan);

    // Get accumulators by aggregating from the input image
    T_acc const accumulator = this->m_aggregator.aggregate(input_src, output_coords, iter_chans);

    // Pointer to the output location
    T_elm_out * p_output = getOutputPointer(output_coords);

    // Apply output transform to accumulators and write to the output image
    this->m_output_transform.transform(p_output, accumulator, output_coords, iter_chans);

  }

}




}}