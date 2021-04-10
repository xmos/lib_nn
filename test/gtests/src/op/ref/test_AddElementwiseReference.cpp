

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "../src/cpp/filt2d/util/TensorWrap.hpp"
#include "ref_tests.hpp"

#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>

/*
  This is just a sanity check for AddElementReference(). It's not meant to thoroughly vet it, because if we're going to
  do that, we have no need for it.
*/

using namespace nn;
using namespace nn::test;


struct AddElementwiseTestParams {
  ImageGeometry image;
  ops::ref::ElementwiseParams op_params;
  int data_seed;
};

inline std::ostream& operator<<(std::ostream &stream, const AddElementwiseTestParams &params){
  return stream << "Image {" << params.image << "}, " 
                << "input[0]{" << params.op_params.input[0].multiplier << "," 
                    << int(params.op_params.input[0].zero_point) << "}, "
                << "input[1]{" << params.op_params.input[1].multiplier << "," 
                    << int(params.op_params.input[1].zero_point) << "}, "
                << "output{"   << params.op_params.output.multiplier   << "," 
                    << int(params.op_params.output.zero_point) << "}, "
                << "data_seed{" << params.data_seed << "}";
}

template <> AddElementwiseTestParams Rand::get_rand<AddElementwiseTestParams>(
    Tag<AddElementwiseTestParams>)
{
  auto res = AddElementwiseTestParams();

  res.image = ImageGeometry(this->rand<unsigned>(1, 10),
                            this->rand<unsigned>(1, 10),
                            this->rand<unsigned>(1, 30));
 
  res.op_params.input[0].zero_point = this->rand<int8_t>();
  res.op_params.input[1].zero_point = this->rand<int8_t>();
  
  res.op_params.input[0].multiplier = this->rand<float>( ldexpf(1, -10), ldexp(3, -10) );
  res.op_params.input[1].multiplier = this->rand<float>( ldexpf(1, -10), ldexp(3, -10) );

  auto min_in0 = (-128 - int(res.op_params.input[0].zero_point)) * res.op_params.input[0].multiplier;
  auto max_in0 = ( 127 - int(res.op_params.input[0].zero_point)) * res.op_params.input[0].multiplier;
  auto min_in1 = (-128 - int(res.op_params.input[1].zero_point)) * res.op_params.input[1].multiplier;
  auto max_in1 = ( 127 - int(res.op_params.input[1].zero_point)) * res.op_params.input[1].multiplier;

  auto min_out = min_in0 + min_in1;
  auto max_out = max_in0 + max_in1;
  auto out_span = max_out - min_out;

  res.op_params.output.multiplier = ldexpf(1, 8) / out_span;

  // -128 = min_out * out_mult + out_zero
  // out_zero = -128 - (min_out * out_mul)
  res.op_params.output.zero_point = std::numeric_limits<int8_t>::min() - (min_out * res.op_params.output.multiplier);

  // res.op_params.output.zero_point   = this->rand<int8_t>();
  // res.op_params.output.multiplier   = this->rand<float>( ldexpf(1, -10), ldexp(3, -10) );

  res.data_seed = this->rand<int>();

  return res;
}

class AddElementwiseReferenceTest : public ::testing::TestWithParam<AddElementwiseTestParams> {};

TEST_P(AddElementwiseReferenceTest, AddElementwiseReference)
{
  auto params = GetParam();

  auto rng = Rand(params.data_seed);

  auto input0   = std::vector<int8_t>( params.image.imageElements() );
  auto input1   = std::vector<int8_t>( params.image.imageElements() );
  auto expected = std::vector<int8_t>( params.image.imageElements() );

  //fill images.
  for(int k = 0; k < input0.size(); k++) {

    float tmp;

    input0[k] = rng.rand<int8_t>();
    input1[k] = rng.rand<int8_t>();

    float v0 = ( int32_t(input0[k]) - params.op_params.input[0].zero_point) * params.op_params.input[0].multiplier;
    float v1 = ( int32_t(input1[k]) - params.op_params.input[1].zero_point) * params.op_params.input[1].multiplier;
    float sum = v0 + v1;
    tmp = (sum / params.op_params.output.multiplier) + params.op_params.output.zero_point;

    expected[k] = round_int8( tmp );
  }

  auto output = ops::ref::AddElementwiseReference(params.image,
                                                  &input0[0], &input1[0],
                                                  params.op_params);

  ASSERT_EQ(output, expected);
}

static auto iterA = nn::test::RandIter<AddElementwiseTestParams>(100, 457645);
INSTANTIATE_TEST_SUITE_P(, AddElementwiseReferenceTest, ::testing::ValuesIn(iterA.begin(), iterA.end()));




