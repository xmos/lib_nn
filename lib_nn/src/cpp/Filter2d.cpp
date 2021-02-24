
#include "Filter2d.hpp"

using namespace nn::filt2d;

template <typename T>
static T const* getPointer(
    T const* base_address,
    PointerCovector const& covector,
    ImageVect const& coords)
{
  int32_t offset = coords.row     * ((int32_t)covector.row_bytes  )
                 + coords.col     * ((int32_t)covector.col_bytes  )
                 + coords.channel * ((int32_t)covector.chan_bytes );
  
  return (T const*) (((int32_t const)base_address) + offset);
}


template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans,
          class T_memcpy,    class T_agg,        class T_ot>
T_elm_in  const* Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::getInputPointer(
    ImageVect const& output_coords) const
{
  return getPointer<T_elm_in>(this->m_input_image, this->m_input_covector, output_coords);
}


template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans,
          class T_memcpy,    class T_agg,        class T_ot>
T_elm_out const* Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::getOutputPointer(
    ImageVect const& output_coords) const
{
  return getPointer<T_elm_out>(this->m_output_image, this->m_output_covector, output_coords);
}


template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans,
          class T_memcpy,    class T_agg,        class T_ot>
void Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::computePixelSlice(
      int const out_row,
      int const out_col,
      int const out_chan_start,
      unsigned const out_chan_count) const
{
  int const last_chan = out_chan_start + out_chan_count;

  auto const* input_src = this->m_mem_copy.copy_mem(
      ImageVect(out_row, out_col, out_chan_start), 
      this->m_input_image, out_chan_count);

  for(int out_chan = out_chan_start; out_chan < last_chan; out_chan += N_cog_chans){

    unsigned const iter_chans = (last_chan - out_chan >= N_cog_chans)? N_cog_chans : (last_chan - out_chan);

    ImageVect const output_coords = ImageVect(out_row, out_col, out_chan);

    T_elm_out const* p_output = getOutputPointer(output_coords);

    T_acc const accumulator = this->m_aggregator.aggregate(input_src, output_coords, iter_chans);

    this->m_output_transform.transform(p_output, accumulator, iter_chans);

  }

}



template <typename T_elm_in, typename T_elm_out, typename T_acc, unsigned N_cog_chans,
          class T_memcpy,    class T_agg,        class T_ot>
void Filter2d<T_elm_in,T_elm_out,T_acc,N_cog_chans,T_memcpy,T_agg,T_ot>::execute(
      ImageRegion const& job) const
{

  for(int row = job.start.row; row < (job.start.row + job.shape.height); row++){
    for(int col = job.start.col; col < (job.start.col + job.shape.width); col++){
      this->computePixelSlice(row, col, job.start.channel, job.shape.depth);
    }
  }

}