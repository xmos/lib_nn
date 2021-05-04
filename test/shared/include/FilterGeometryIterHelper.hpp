#pragma once

#include "FilterGeometryIter.hpp"

namespace nn {
  namespace test {

    using namespace nn::ff;


    namespace unpadded {

      /**
       * Iterates over output image shapes and window shapes with:
       *  - Window start = (0,0)
       *  - Window spatial strides = window spatial shape
       *  - Dilation = (0,0)
       * The input image's geometry will be calculated based on the output image geometry
       * and the window geometry such that the entire input image is consumed and no padding
       * is used.
       *
       */
      static inline nn::ff::FilterGeometryIterator SimpleDepthwise( const std::array<int,2> output_spatial_range,
                                                                    const std::array<int,2> window_spatial_range,
                                                                    const std::array<int,2> channel_count_range,
                                                                    const int output_step = 1,
                                                                    const int window_step = 1,
                                                                    const int channel_step = 4)
      {
        return FilterGeometryIterator( 
                  nn::Filter2dGeometry( {output_spatial_range[0], output_spatial_range[0], channel_count_range[0]},   
                                        {1, 1, 4}, 
                                        { {window_spatial_range[0], window_spatial_range[0], 1}, {0, 0}, {0, 0, 1}, {1, 1} } ), 
                  {
                    new OutputShape( {output_spatial_range[0], output_spatial_range[0], channel_count_range[0]}, 
                                     {output_spatial_range[1], output_spatial_range[1], channel_count_range[1]}, 
                                     {output_step, output_step, channel_step} ),
                    new WindowShape( {window_spatial_range[0], window_spatial_range[0]},
                                     {window_spatial_range[1], window_spatial_range[1]},
                                     {window_step, window_step} ),
                    new Apply( nn::ff::MakeUnpadded ) } );
      }

      
    }





  }
}