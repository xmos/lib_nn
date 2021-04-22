
#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>
#include <cassert>
#include <limits>

#include "MaxPool2d.hpp"

#include "gtest/gtest.h"

#include "AggregateFn.hpp"
#include "Rand.hpp"
#include "VpuHelpers.hpp"
#include "RefOps.hpp"

static constexpr bool DETAILED_FAILURE_OUTPUT = true;

// Need different test classes for each of the operators only because they use Filter2dGeometry
// as a test parameter and some of the geometries don't apply to some MaxPool2d operators, and
// gtest doesn't give us a way of specifying that.

class MaxPool2d_Generic_Test : public ::testing::TestWithParam<nn::Filter2dGeometry> {};
class MaxPool2d_Valid_Test : public ::testing::TestWithParam<nn::Filter2dGeometry> {};

/**
 * Generates a random input image
 */
static std::vector<int8_t> GenerateInputImage(
    const nn::Filter2dGeometry& filter,
    nn::test::Rand& rnd)
{
  std::vector<int8_t> input_img( filter.input.imageElements() );
  rnd.rand_bytes( &input_img[0], input_img.size() * sizeof(int8_t) );
  return input_img;
}

/**
 * Construct and execute a MaxPool2d_Generic operator with the specified geometry using the
 * provided inputs and return the output image.
 */
static std::vector<int8_t> RunOperator_Generic(
    const nn::Filter2dGeometry& filter,
    const nn::ImageRegion& region,
    const std::vector<int8_t> input_img) 
{
  auto output_img = std::vector<int8_t>( filter.output.imageElements() );

  std:memset(&output_img[0], 0, output_img.size() * sizeof(int8_t));

  nn::MaxPool2d_Generic::Params params( filter, region );
  
  nn::ImToColPadded memcopy_handler( &params.mem_params );
  nn::MaxPoolPatchFn agg_handler( &params.agg_params );
  nn::DirectWriteOutputTransform ot_handler( &params.ot_params );

  auto scratch_mem = std::vector<int8_t>( memcopy_handler.get_scratch_bytes() );

  nn::MaxPool2d_Generic mp_op( &params.ak_params, 
                               &memcopy_handler, 
                               &agg_handler,
                               &ot_handler,
                               &scratch_mem[0] );
  
  mp_op.execute(&output_img[0], const_cast<int8_t*>(&input_img[0]));

  return output_img;
}


/**
 * Construct and execute a MaxPool2d_Valid operator with the specified geometry using the
 * provided inputs and return the output image.
 */
static std::vector<int8_t> RunOperator_Valid(
    const nn::Filter2dGeometry& filter,
    const nn::ImageRegion& region,
    const std::vector<int8_t> input_img) 
{
  auto output_img = std::vector<int8_t>( filter.output.imageElements() );

  std:memset(&output_img[0], 0, output_img.size() * sizeof(int8_t));

  nn::MaxPool2d_Valid::Params params( filter, region );
  
  nn::DerefInputFn memcopy_handler( &params.mem_params );
  nn::MaxPoolDirectValidFn agg_handler( &params.agg_params );
  nn::DirectWriteOutputTransform ot_handler( &params.ot_params );

  nn::MaxPool2d_Valid mp_op( &params.ak_params, 
                             &memcopy_handler, 
                             &agg_handler,
                             &ot_handler );
  
  mp_op.execute(&output_img[0], const_cast<int8_t*>(&input_img[0]));

  return output_img;
}




/**
 * Test Case -- Compare output of  MaxPool2d_Generic to our reference implementation
 */
TEST_P(MaxPool2d_Generic_Test, CompareWithReference)
{
  constexpr int REPETITIONS = 2;

  const auto filter = GetParam();

  auto rand = nn::test::Rand(  (filter.input.imageBytes()+7) 
                             * (filter.window.shape.imageBytes()+13)
                             * (filter.output.imageBytes()+23) );

  const auto cog_count = nn::MaxPool2d::OutputGroups( filter.output.depth );

  for(int iter = 0; iter < REPETITIONS; iter++){
    auto input_img = GenerateInputImage(filter, rand);

    // const auto input_zero_point = rand.rand<int8_t>(-20, 20);

    const auto cog_start = rand.rand<int>( 0, cog_count - 1);
                                    
    auto ref_output = nn::test::ops::ref::MaxPoolReference(filter, &input_img[0]);

    auto op_output = RunOperator_Generic(filter, filter.GetFullJob(), input_img);

    // This is really unfortunate, but computing all the string stuff below for every output element 
    // REALLY slows everything down (by like an order of magnitude) even when the test case doesn't fail.
    // Of course the Fold() operation below is expensive, but even without that this is quite expensive.
    // This is ugly, but it lets it run fast without sacrificing the useful output.
    if(DETAILED_FAILURE_OUTPUT){
      for(int row = 0; row < filter.output.height; row++){
        for(int col = 0; col < filter.output.width; col++){
          for(int chan = 0; chan < filter.output.depth; chan++){
            const auto offset = (row * filter.output.width + col) * filter.output.depth + chan;

            // This is quite unfortunate, but if I have to evaluate all this stream stuff
            if( ref_output[offset] != op_output[offset] ){

              auto loc = filter.GetWindow(nn::ImageVect(row, col, chan));
              std::stringstream stream;
              stream << "max([ ";

              auto mx = loc.Fold<int8_t,int8_t>(&input_img[0], 
                            [&](const nn::ImageVect& filter_coords, const nn::ImageVect& input_coords,
                                const int8_t acc, const int8_t element, const bool is_padding) -> int8_t 
                            {
                              stream << int(element) << ", ";
                              return std::max<int8_t>(acc, element);
                            }, INT8_MIN, INT8_MIN);
              stream << "]) = " << int(mx);

              ASSERT_EQ(ref_output[offset], op_output[offset]) << "Failure Details:\n"
                  << "\t| iter = " << iter << "\n"
                  << "\t| offset = " << offset << "\n"
                  << "\t| ref_output[" << row << "][" << col << "][" << chan << "] = " << int(ref_output[offset]) << "\n"
                  << "\t|  op_output[" << row << "][" << col << "][" << chan << "] = " << int(op_output[offset]) << "\n"
                  // << "\t| input_zero_point = " << int(input_zero_point) << "\n"
                  << "\t| " << stream.str();
            }
          }
        }
      }
    }

    ASSERT_EQ(ref_output, op_output);
  }
}

/**
 * Test Case -- Compare output of  MaxPool2d_Valid to our reference implementation
 */
TEST_P(MaxPool2d_Valid_Test, CompareWithReference)
{
  constexpr int REPETITIONS = 2;

  const auto filter = GetParam();

  auto rand = nn::test::Rand(  (filter.input.imageBytes()+7) 
                             * (filter.window.shape.imageBytes()+13)
                             * (filter.output.imageBytes()+23) );

  const auto cog_count = nn::MaxPool2d::OutputGroups( filter.output.depth );

  for(int iter = 0; iter < REPETITIONS; iter++){
    auto input_img = GenerateInputImage(filter, rand);

    const auto input_zero_point = rand.rand<int8_t>(-20, 20);

    const auto cog_start = rand.rand<int>( 0, cog_count - 1);
                                    
    auto ref_output = nn::test::ops::ref::MaxPoolReference(filter, &input_img[0]);

    auto op_output = RunOperator_Valid(filter, filter.GetFullJob(), input_img);

    // This is really unfortunate, but computing all the string stuff below for every output element 
    // REALLY slows everything down (by like an order of magnitude) even when the test case doesn't fail.
    // Of course the Fold() operation below is expensive, but even without that this is quite expensive.
    // This is ugly, but it lets it run fast without sacrificing the useful output.
    if(DETAILED_FAILURE_OUTPUT){
      for(int row = 0; row < filter.output.height; row++){
        for(int col = 0; col < filter.output.width; col++){
          for(int chan = 0; chan < filter.output.depth; chan++){
            const auto offset = (row * filter.output.width + col) * filter.output.depth + chan;

            // This is quite unfortunate, but if I have to evaluate all this stream stuff
            if( ref_output[offset] != op_output[offset] ){

              auto loc = filter.GetWindow(nn::ImageVect(row, col, chan));
              std::stringstream stream;
              stream << "max([ ";

              auto mx = loc.Fold<int8_t,int8_t>(&input_img[0], 
                            [&](const nn::ImageVect& filter_coords, const nn::ImageVect& input_coords,
                                const int8_t acc, const int8_t element, const bool is_padding) -> int8_t 
                            {
                              stream << int(element) << ", ";
                              return std::max<int8_t>(acc, element);
                            }, INT8_MIN, input_zero_point);
              stream << "]) = " << int(mx);

              ASSERT_EQ(ref_output[offset], op_output[offset]) << "Failure Details:\n"
                  << "\t| iter = " << iter << "\n"
                  << "\t| offset = " << offset << "\n"
                  << "\t| ref_output[" << row << "][" << col << "][" << chan << "] = " << int(ref_output[offset]) << "\n"
                  << "\t|  op_output[" << row << "][" << col << "][" << chan << "] = " << int(op_output[offset]) << "\n"
                  << "\t| " << stream.str();
            }
          }
        }
      }
    }

    ASSERT_EQ(ref_output, op_output);
  }
}


/**
 * Generate a sequence of Filter2dGeometries that do not use padding or dilations other than 1.
 */
static std::vector<nn::Filter2dGeometry> NoPaddingFilters()
{
  constexpr int MAX_Y_DIM =   8;
  constexpr int MAX_Y_CHANS = 36;
  constexpr int MAX_K_DIM =   4;
  auto vec = std::vector<nn::Filter2dGeometry>();

  for(int Y_h = 1; Y_h <= MAX_Y_DIM; ++Y_h){
    for(int Y_w = 1; Y_w <= MAX_Y_DIM; ++Y_w){
      for(int Y_d = 4; Y_d <= MAX_Y_CHANS; Y_d+=4){

        nn::ImageGeometry output_img(Y_h, Y_w, Y_d);

        for(int K_h = 1; K_h <= MAX_K_DIM; ++K_h){
          for(int K_w = 1; K_w <= MAX_K_DIM; ++K_w){

            nn::WindowGeometry window(K_h, K_w, 1,  //Shape
                                      0, 0,         //Start
                                      K_h, K_w, 1,  //Stride
                                      1, 1);        //Dilation

            nn::ImageGeometry input_img( Y_h * K_h, Y_w * K_w, Y_d );
            vec.push_back( nn::Filter2dGeometry(input_img, output_img, window) );

  } } } } }

  return vec;
}

/**
 * Generate a sequence of Filter2dGeometries that do use padding, but no dilation.
 * 
 * This will still be relatively simple. Padding will be added to the top/left by just
 * setting the window start to (1-K_h) and (1-K_w). Padding will then be added to the
 * bottom/right by subtracting K_h and K_w from the input dimensions.
 */
static std::vector<nn::Filter2dGeometry> PaddedFilters()
{
  constexpr int MAX_Y_DIM =   8;
  constexpr int MAX_Y_CHANS = 36;
  constexpr int MAX_K_DIM =   4;
  auto vec = std::vector<nn::Filter2dGeometry>();

  for(int Y_h = 1; Y_h <= MAX_Y_DIM; ++Y_h){
    for(int Y_w = 1; Y_w <= MAX_Y_DIM; ++Y_w){
      for(int Y_d = 4; Y_d <= MAX_Y_CHANS; Y_d+=4){

        nn::ImageGeometry output_img(Y_h, Y_w, Y_d);

        for(int K_h = 2; K_h <= MAX_K_DIM; ++K_h){
          for(int K_w = 2; K_w <= MAX_K_DIM; ++K_w){
            
            nn::ImageGeometry input_img( (Y_h-1) * K_h, (Y_w-1) * K_w, Y_d );

            if(input_img.height < 1 || input_img.width < 1) continue;

            nn::WindowGeometry window(K_h, K_w, 1,  //Shape
                                      1-K_h, 1-K_w, //Start
                                      K_h, K_w, 1,  //Stride
                                      1, 1);        //Dilation

            vec.push_back( nn::Filter2dGeometry(input_img, output_img, window) );

  } } } } }

  return vec;
}

/////////////////////////////////
/////////////////////////////////

static auto no_pad_iter = ::testing::ValuesIn( NoPaddingFilters() );
static auto padded_iter = ::testing::ValuesIn( PaddedFilters() );

INSTANTIATE_TEST_SUITE_P(Unpadded, MaxPool2d_Generic_Test, no_pad_iter);
INSTANTIATE_TEST_SUITE_P(Unpadded, MaxPool2d_Valid_Test, no_pad_iter);
INSTANTIATE_TEST_SUITE_P(Padded, MaxPool2d_Generic_Test, padded_iter);