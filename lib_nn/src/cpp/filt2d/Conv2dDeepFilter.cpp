
#include "Conv2dDeepFilter.hpp"
#include "xs3_vpu.h"

#include <memory>
#include <vector>
#include <limits>
#include <cmath>
#include <iostream>

using namespace nn::filt2d::op;



void Conv2dDeepFilter_Valid::Job::computeSlice(
    const ImageVect& slice_coords, 
    const unsigned out_chan_count) 
{

  // Copy the intersection of the input image and the convolution window into the patch memory, 
  // filling in with padding as needed.
  auto const* input_src = handler.patch.copy_patch(slice_coords, filter.image.input, out_chan_count);

  // Iterate over channels of the output pixel, handling N_max_cog_chans channels per iteration (except possibly the last)
  for(int out_chan = 0; out_chan < out_chan_count; out_chan += N_max_cog_chans){

    ImageVect const output_coords = slice_coords.add(0,0,out_chan);
    
    // Output channels for this iteration
    const unsigned iter_chans = std::min(N_max_cog_chans, out_chan_count - out_chan);

    // Get accumulators by aggregating from the input image
    vpu_split_acc32_t const accumulator = handler.agg.aggregate(input_src, output_coords, iter_chans);

    // Pointer to the output location
    auto * p_output = filter.config.output.covector.resolve(filter.image.output, output_coords);

    // Apply output transform to accumulators and write to the output image
    handler.ot.transform(p_output, accumulator, output_coords, iter_chans);

  }
}


Conv2dDeepFilter_Valid::JobClass Conv2dDeepFilter_Valid::spawnJob(
    const ImageRegion& region,
    T_elm_in* patch_mem) const
{
  return JobClass(
            Job(*this, 
                T_patch(this->config.handler.patch, patch_mem),
                T_agg(this->config.handler.agg), 
                T_ot(this->config.handler.ot)), 
            region);
}


void Conv2dDeepFilter_Valid::execute(T_elm_in* patch_mem) const
{
  this->spawnJob(
    ImageRegion(0,0,0,config.output.shape.height, config.output.shape.width, config.output.shape.depth),
    patch_mem).execute();
}


////////////////////////////////////////////////////////
/////
////////////////////////////////////////////////////////

void Conv2dDeepFilter::Job::computeSlice(
    const ImageVect& slice_coords, 
    const unsigned out_chan_count) 
{

  // Copy the intersection of the input image and the convolution window into the patch memory, 
  // filling in with padding as needed.
  auto const* input_src = handler.patch.copy_patch(slice_coords, filter.image.input, out_chan_count);

  // Iterate over channels of the output pixel, handling N_max_cog_chans channels per iteration (except possibly the last)
  for(int out_chan = 0; out_chan < out_chan_count; out_chan += N_max_cog_chans){

    ImageVect const output_coords = slice_coords.add(0,0,out_chan);
    
    // Output channels for this iteration
    unsigned const iter_chans = std::min(N_max_cog_chans, out_chan_count - out_chan);

    // Get accumulators by aggregating from the input image
    vpu_split_acc32_t const accumulator = handler.agg.aggregate(input_src, output_coords, iter_chans);

    // Pointer to the output location
    auto * p_output = filter.config.output.covector.resolve(filter.image.output, output_coords);

    // Apply output transform to accumulators and write to the output image
    handler.ot.transform(p_output, accumulator, output_coords, iter_chans);

  }
}


Conv2dDeepFilter::JobClass Conv2dDeepFilter::spawnJob(
    const ImageRegion& region,
    T_elm_in* patch_mem) const
{
  return JobClass(
            Job(*this, 
                T_patch(this->config.handler.patch, patch_mem),
                T_agg(this->config.handler.agg), 
                T_ot(this->config.handler.ot)), 
            region);
}


void Conv2dDeepFilter::execute(T_elm_in* patch_mem) const
{
  this->spawnJob(
    ImageRegion(0,0,0,config.output.shape.height, config.output.shape.width, config.output.shape.depth),
    patch_mem).execute();
}

