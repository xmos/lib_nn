
#include <cassert>
#include <iostream>
#include <limits>
#include <sstream>
#include <tuple>
#include <vector>

#include "AggregateFn.hpp"
#include "FilterGeometryIter.hpp"
#include "MaxPool2d.hpp"
#include "Rand.hpp"
#include "RefOps.hpp"
#include "VpuHelpers.hpp"
#include "gtest/gtest.h"

static constexpr bool DETAILED_FAILURE_OUTPUT = true;

/**
 * Generates a random input image
 */
static std::vector<int8_t> GenerateInputImage(
    const nn::Filter2dGeometry& filter, nn::test::Rand& rnd) {
  std::vector<int8_t> input_img(filter.input.ElementCount());
  rnd.rand_bytes(&input_img[0], input_img.size() * sizeof(int8_t));
  return input_img;
}

/**
 * Construct and execute a MaxPool2d_Generic operator with the specified
 * geometry using the provided inputs and return the output image.
 */
static std::vector<int8_t> RunOperator_Generic(
    const nn::Filter2dGeometry& filter, const nn::ImageRegion& region,
    const std::vector<int8_t> input_img) {
  auto output_img = std::vector<int8_t>(filter.output.ElementCount());

  std::memset(&output_img[0], 0, output_img.size() * sizeof(int8_t));

  nn::MaxPool2d_Generic::Params params(filter, region);

  nn::ImToColPadded memcopy_handler(&params.mem_params);
  nn::MaxPoolPatchFn agg_handler(&params.agg_params);
  nn::DirectWriteOutputTransform ot_handler(&params.ot_params);

  auto scratch_mem = std::vector<int8_t>(memcopy_handler.get_scratch_bytes());

  nn::MaxPool2d_Generic mp_op(&params.ak_params, &memcopy_handler, &agg_handler,
                              &ot_handler);

  mp_op.execute(&output_img[0], const_cast<int8_t*>(&input_img[0]));

  return output_img;
}

/**
 * Construct and execute a MaxPool2d_Valid operator with the specified geometry
 * using the provided inputs and return the output image.
 */
static std::vector<int8_t> RunOperator_Valid(
    const nn::Filter2dGeometry& filter, const nn::ImageRegion& region,
    const std::vector<int8_t> input_img) {
  auto output_img = std::vector<int8_t>(filter.output.ElementCount());

  std::memset(&output_img[0], 0, output_img.size() * sizeof(int8_t));

  nn::MaxPool2d_Valid::Params params(filter, region);

  nn::DerefInputFn memcopy_handler(&params.mem_params);
  nn::MaxPoolDirectValidFn agg_handler(&params.agg_params);
  nn::DirectWriteOutputTransform ot_handler(&params.ot_params);

  nn::MaxPool2d_Valid mp_op(&params.ak_params, &memcopy_handler, &agg_handler,
                            &ot_handler);

  mp_op.execute(&output_img[0], const_cast<int8_t*>(&input_img[0]));

  return output_img;
}

static nn::ff::FilterGeometryIterator TestFilterIterator() {
  using namespace nn::ff;
  return FilterGeometryIterator(
      nn::Filter2dGeometry({1, 1, 1}, {1, 1, 4},
                           {{1, 1, 1}, {0, 0}, {1, 1, 1}, {1, 1}}),
      {new OutputShape({1, 1, 4}, {4, 8, 36}, {1, 1, 4}),
       new WindowShape({1, 1}, {4, 4}, {1, 1}),
       new FrameSequence({new Apply(MakeUnpaddedDepthwise),
                          new Apply(MakePaddedDepthwise)})});
}

/**
 * Test Case -- Compare output of  MaxPool2d_Generic to our reference
 * implementation
 */
TEST(MaxPool2d_Generic_Test, CompareWithReference) {
  /*
  constexpr int REPETITIONS = 2;
  int total_iters = 0;

  for (auto filter : TestFilterIterator()) {
    if (!nn::MaxPool2d_Generic::SupportsGeometry(filter)) {
      // std::cout << "rejected filter: " << filter << std::endl;
      continue;
    }

    auto rand = nn::test::Rand((filter.input.ImageBytes() + 7) *
                               (filter.window.shape.ImageBytes() + 13) *
                               (filter.output.ImageBytes() + 23));

    for (int iter = 0; iter < REPETITIONS; iter++) {
      auto input_img = GenerateInputImage(filter, rand);

      // const auto input_zero_point = rand.rand<int8_t>(-20, 20);

      auto ref_output =
          nn::test::ops::ref::MaxPoolReference(filter, &input_img[0]);

      auto full_job = nn::ImageRegion(0, 0, 0, filter.output.height,
                                      filter.output.width, filter.output.depth);
      auto op_output = RunOperator_Generic(filter, full_job, input_img);

      // This is really unfortunate, but computing all the string stuff below
      // for every output element REALLY slows everything down (by like an order
      // of magnitude) even when the test case doesn't fail. Of course the
      // Fold() operation below is expensive, but even without that this is
      // quite expensive. This is ugly, but it lets it run fast without
      // sacrificing the useful output.
      if (DETAILED_FAILURE_OUTPUT) {
        for (int row = 0; row < filter.output.height; row++) {
          for (int col = 0; col < filter.output.width; col++) {
            for (int chan = 0; chan < filter.output.depth; chan++) {
              const auto offset =
                  (row * filter.output.width + col) * filter.output.depth +
                  chan;

              // This is quite unfortunate, but if I have to evaluate all this
              // stream stuff
              if (ref_output[offset] != op_output[offset]) {
                auto loc = filter.GetWindow(nn::ImageVect(row, col, chan));
                std::stringstream stream;
                stream << "max([ ";

                auto mx = loc.Fold<int8_t, int8_t>(
                    &input_img[0],
                    [&](const nn::ImageVect& filter_coords,
                        const nn::ImageVect& input_coords, const int8_t acc,
                        const int8_t element, const bool is_padding) -> int8_t {
                      stream << int(element) << ", ";
                      return std::max<int8_t>(acc, element);
                    },
                    INT8_MIN, INT8_MIN);
                stream << "]) = " << int(mx);

                ASSERT_EQ(ref_output[offset], op_output[offset])
                    << "Failure Details:\n"
                    << "\t| iter = " << iter << "\n"
                    << "\t| offset = " << offset << "\n"
                    << "\t| ref_output[" << row << "][" << col << "][" << chan
                    << "] = " << int(ref_output[offset]) << "\n"
                    << "\t|  op_output[" << row << "][" << col << "][" << chan
                    << "] = " << int(op_output[offset])
                    << "\n"
                    // << "\t| input_zero_point = " << int(input_zero_point) <<
                    // "\n"
                    << "\t| " << stream.str();
              }
            }
          }
        }
      }

      ASSERT_EQ(ref_output, op_output);
      total_iters++;
    }
  }

  std::cout << "Count: " << total_iters << std::endl;
  */
}

/**
 * Test Case -- Compare output of  MaxPool2d_Valid to our reference
 * implementation
 */
TEST(MaxPool2d_Valid_Test, CompareWithReference) {
  /*
  constexpr int REPETITIONS = 2;
  int total_iters = 0;

  for (auto filter : TestFilterIterator()) {
    if (!nn::MaxPool2d_Valid::SupportsGeometry(filter)) {
      // std::cout << "rejected filter: " << filter << std::endl;
      continue;
    }

    auto rand = nn::test::Rand((filter.input.ImageBytes() + 7) *
                               (filter.window.shape.ImageBytes() + 13) *
                               (filter.output.ImageBytes() + 23));

    const auto cog_count = nn::MaxPool2d::OutputGroups(filter.output.depth);

    for (int iter = 0; iter < REPETITIONS; iter++) {
      auto input_img = GenerateInputImage(filter, rand);

      const auto input_zero_point = rand.rand<int8_t>(-20, 20);

      const auto cog_start = rand.rand<int>(0, cog_count - 1);

      auto ref_output =
          nn::test::ops::ref::MaxPoolReference(filter, &input_img[0]);

      auto full_job = nn::ImageRegion(0, 0, 0, filter.output.height,
                                      filter.output.width, filter.output.depth);
      auto op_output = RunOperator_Valid(filter, full_job, input_img);

      // This is really unfortunate, but computing all the string stuff below
      // for every output element REALLY slows everything down (by like an order
      // of magnitude) even when the test case doesn't fail. Of course the
      // Fold() operation below is expensive, but even without that this is
      // quite expensive. This is ugly, but it lets it run fast without
      // sacrificing the useful output.
      if (DETAILED_FAILURE_OUTPUT) {
        for (int row = 0; row < filter.output.height; row++) {
          for (int col = 0; col < filter.output.width; col++) {
            for (int chan = 0; chan < filter.output.depth; chan++) {
              const auto offset =
                  (row * filter.output.width + col) * filter.output.depth +
                  chan;

              // This is quite unfortunate, but if I have to evaluate all this
              // stream stuff
              if (ref_output[offset] != op_output[offset]) {
                auto loc = filter.GetWindow(nn::ImageVect(row, col, chan));
                std::stringstream stream;
                stream << "max([ ";

                auto mx = loc.Fold<int8_t, int8_t>(
                    &input_img[0],
                    [&](const nn::ImageVect& filter_coords,
                        const nn::ImageVect& input_coords, const int8_t acc,
                        const int8_t element, const bool is_padding) -> int8_t {
                      stream << int(element) << ", ";
                      return std::max<int8_t>(acc, element);
                    },
                    INT8_MIN, input_zero_point);
                stream << "]) = " << int(mx);

                ASSERT_EQ(ref_output[offset], op_output[offset])
                    << "Failure Details:\n"
                    << "\t| iter = " << iter << "\n"
                    << "\t| offset = " << offset << "\n"
                    << "\t| ref_output[" << row << "][" << col << "][" << chan
                    << "] = " << int(ref_output[offset]) << "\n"
                    << "\t|  op_output[" << row << "][" << col << "][" << chan
                    << "] = " << int(op_output[offset]) << "\n"
                    << "\t| " << stream.str();
              }
            }
          }
        }
      }

      ASSERT_EQ(ref_output, op_output);
      total_iters++;
    }
  }

  std::cout << "Count: " << total_iters << std::endl;
  */
}
