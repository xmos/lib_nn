

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/util/conv2d_utils.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "../src/cpp/filt2d/Conv2dDeepFilter.hpp"
#include "NNOps.hpp"
#include "RefOps.hpp"

#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>

using namespace nn;
using namespace nn::op;



class Conv2dDeepFilter_ValidTest : public ::testing::TestWithParam<Filter2dGeometry> {
  
  private: 

    // Private because they should be accessed through GetNNOutput() and GetRefOutput(),
    // which actually populate them by running the operator.
    std::vector<int8_t> nn_output;
    std::vector<int8_t> ref_output;

  protected:

    std::vector<int8_t> input_image;
    std::vector<int8_t> kernel_tensor;
    std::vector<int32_t> bias;
    std::vector<float> output_multiplier;

    int8_t input_zero_point;
    int8_t output_zero_point;

    /**
     * Initialize our inputs and parameters with vectors of the appropriate size.
     */
    virtual void SetUp() override {
      auto geometry = this->GetParam();

      const auto out_chans = geometry.output.depth;

      input_image.resize(geometry.input.imageBytes());
      kernel_tensor.resize(geometry.window.shape.imageBytes() * out_chans);
      bias.resize(out_chans);
      output_multiplier.resize(out_chans);
      nn_output.clear();
      ref_output.clear();

      memset(&input_image[0], 0, sizeof(int8_t) * input_image.size());
      memset(&kernel_tensor[0], 0, sizeof(int8_t) * kernel_tensor.size());
      memset(&bias[0], 0, sizeof(int32_t) * bias.size());
      memset(&output_multiplier[0], 0, sizeof(float) * output_multiplier.size());

      input_zero_point = 0;
      output_zero_point = 0;
    }

    /**
     * Get the lib_nn output. 
     * If it hasn't been computed yet, do so here. Otherwise, just return the previous output.
     */
    virtual std::vector<int8_t>& GetNNOutput() {
      auto geometry = this->GetParam();
      if(this->nn_output.size() == 0)
        this->nn_output = nn::test::ops::Conv2dDeepFilter_Valid( 
                                    geometry, &input_image[0], &kernel_tensor[0], 
                                    &bias[0], &output_multiplier[0], 
                                    input_zero_point, output_zero_point);
      // I wanted this to be an ASSERT_GT(), but that macro ultimately attempts to return from the test, so it
      // doesn't work here.
      EXPECT_GT(this->nn_output.size(), 0);
      return this->nn_output;
    }

    /**
     * Get the ref output. 
     * If it hasn't been computed yet, do so here. Otherwise, just return the previous output.
     */
    virtual std::vector<int8_t>& GetRefOutput() {
      auto geometry = this->GetParam();
      if(this->ref_output.size() == 0)
        this->ref_output = nn::test::ops::ref::Conv2dDenseReference(
                                    geometry, &input_image[0], &kernel_tensor[0], 
                                    &bias[0], &output_multiplier[0], 
                                    input_zero_point, output_zero_point);
      // I wanted this to be an ASSERT_GT(), but that macro ultimately attempts to return from the test, so it
      // doesn't work here.
      EXPECT_GT(this->ref_output.size(), 0);
      return this->ref_output;
    }

    /**
     * Execute the lib_nn and reference implementations for the operator (if they haven't already been executed)
     * and compare their results.
     */
    virtual void ExecuteAndCompare() {
      auto geometry = this->GetParam();
      auto nn_out = this->GetNNOutput();
      auto ref_out = this->GetRefOutput();

      ASSERT_EQ(nn_out.size(), geometry.output.imageElements())
        << "lib_nn output was not the expected size.";
      ASSERT_EQ(ref_out.size(), geometry.output.imageElements())
        << "reference output was not the expected size.";

      ASSERT_EQ(nn_out, ref_out)
        << "lib_nn output did not match reference output with geometry: " << geometry;
    }
    
};

/**
 * Uses 0 for all parameters. Just to make sure no exceptions happen.
 */
TEST_P(Conv2dDeepFilter_ValidTest,RunsWithoutException)
{
  auto geometry = GetParam();

  memset(&input_image[0], 0, sizeof(int8_t) * input_image.size());
  memset(&kernel_tensor[0], 0, sizeof(int8_t) * kernel_tensor.size());

  for(int i = 0; i < geometry.output.depth; ++i){
    bias[i] = 0;
    output_multiplier[i] = 0.0f;
  }

  int8_t expected_out = 0;

  ASSERT_NO_FATAL_FAILURE(ExecuteAndCompare());

  ASSERT_EQ(GetNNOutput()[0], expected_out);

}

/**
 * Uses 1 for input and kernel weights. Sets the bias so that all accumulators end up as 27.
 * 
 * Checks that the accumulation is correct in this simple case.
 */
TEST_P(Conv2dDeepFilter_ValidTest,InputAndWeightsAre1)
{
  auto geometry = GetParam();

  auto window_elements = geometry.window.shape.imageElements();

  memset(&input_image[0], 1, sizeof(int8_t) * input_image.size());
  memset(&kernel_tensor[0], 1, sizeof(int8_t) * kernel_tensor.size());

  for(int i = 0; i < geometry.output.depth; ++i){
    bias[i] = -window_elements + 27;
    output_multiplier[i] = 1.0f;
  }

  int8_t expected_out = 27;

  ASSERT_NO_FATAL_FAILURE(ExecuteAndCompare());

  ASSERT_EQ(GetNNOutput()[0], expected_out);

}

INSTANTIATE_TEST_SUITE_P(, Conv2dDeepFilter_ValidTest, 
  ::testing::ValuesIn(Conv2dDeepFilter_Valid::GetGeometryIterator().begin(),
                      Conv2dDeepFilter_Valid::GetGeometryIterator().end()));
