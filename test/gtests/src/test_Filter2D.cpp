#include <cstring>
#include <vector>

#include "Filter2D.hpp"
#include "Rand.hpp"
#include "gtest/gtest.h"

namespace nn {

class MockMemCpyFn : public MemCpyFn {
 public:
  struct memcopy_fn_call {
    int8_t *T, *X;
    int32_t h, w, c;
  };

  struct {
    std::vector<memcopy_fn_call> memcopy_fn;
    int get_scratch_bytes;
    int get_overread_bytes;
  } calls;

 public:
  MockMemCpyFn() : calls{std::vector<memcopy_fn_call>(0), 0, 0} {}
  int8_t *memcopy_fn(int8_t *T, int8_t *X, int32_t h, int32_t w, int32_t c) {
    this->calls.memcopy_fn.push_back(memcopy_fn_call{T, X, h, w, c});
    return T;
  }
  int get_scratch_bytes() {
    this->calls.get_scratch_bytes++;
    return 0;
  }
  int get_overread_bytes() {
    this->calls.get_overread_bytes++;
    return 0;
  }
};

class MockAggregateFn : public AggregateFn {
 public:
  struct aggregate_fn_call {
    VPURingBuffer *A;
    int8_t *T;
    int32_t output_channel_group;
  };

  struct {
    std::vector<aggregate_fn_call> aggregate_fn;
  } calls;

  MockAggregateFn() : calls{std::vector<aggregate_fn_call>()} {}

  void aggregate_fn(VPURingBuffer *A, int8_t *T, int32_t output_channel_group) {
    this->calls.aggregate_fn.push_back({A, T, output_channel_group});
  }
};

class MockOutputTransform : public OutputTransformFn {
 public:
  struct output_transform_fn_call {
    int8_t *Y;
    VPURingBuffer *A;
    int32_t output_channel_group;
  };

 private:
  int32_t output_slice_channel_count;

 public:
  struct {
    std::vector<output_transform_fn_call> output_transform_fn;
  } calls;

  MockOutputTransform(int32_t output_slice_channel_count)
      : output_slice_channel_count(output_slice_channel_count),
        calls{std::vector<output_transform_fn_call>(0)} {}

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group) {
    this->calls.output_transform_fn.push_back({Y, A, output_channel_group});
    int output_count = std::min<int>(
        output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
        VPU_INT16_EPV);
    for (int ch = 0; ch < output_count; ++ch) Y[ch] = 1;
    return Y + output_count;
  }
};

template <typename T>
class Filter2D_Test : public ::testing::Test {};

using Filter2D_Test_Types = ::testing::Types<Filter2D, Filter2D_DW>;
TYPED_TEST_SUITE(Filter2D_Test, Filter2D_Test_Types);

TYPED_TEST(Filter2D_Test, BasicTest) {
  const auto mult_memcopy = TypeParam::UsesPerGroupMemCopy;

  for (int y_height = 1; y_height <= 8; ++y_height) {
    for (int y_width = 1; y_width <= 8; ++y_width) {
      for (int y_channels = 1; y_channels <= 8; y_channels += 1) {
        for (int r_height_start = 0; r_height_start < y_height;
             ++r_height_start) {
          for (int r_height_end = r_height_start + 1; r_height_end <= y_height;
               ++r_height_end) {
            for (int r_width_start = 0; r_width_start < y_width;
                 ++r_width_start) {
              for (int r_width_end = r_width_start + 1; r_width_end <= y_width;
                   ++r_width_end) {
                for (int r_channels_start = 0; r_channels_start < y_channels;
                     ++r_channels_start) {
                  for (int r_channels_end = r_channels_start + 1;
                       r_channels_end <= y_channels; ++r_channels_end) {
                    int const cog_size = VPU_INT8_ACC_PERIOD;

                    auto ir = ImageRegion(r_height_start, r_width_start,
                                          r_channels_start,
                                          r_height_end - r_height_start,
                                          r_width_end - r_width_start,
                                          r_channels_end - r_channels_start);

                    auto ip = ImageGeometry(y_height, y_width, y_channels);

                    const auto region_pixels = ir.PixelCount();
                    const auto cog_count = ir.ChannelOutputGroups(cog_size);

                    MockAggregateFn agg_fn;
                    MockMemCpyFn mem_fn;
                    MockOutputTransform ot_fn(r_channels_end -
                                              r_channels_start);

                    auto akp =
                        typename AbstractKernel::Params(ip, ir, cog_size);

                    TypeParam f(&akp, &mem_fn, &agg_fn, &ot_fn);

                    int8_t Y[y_height][y_width][y_channels];

                    std::memset(Y, 0, sizeof(Y));

                    f.execute((int8_t *)Y, nullptr);

#define ITER_MSG "Y: " << ip << " | Output Region: " << ir

                    const auto expected_memcopy_calls =
                        region_pixels * (mult_memcopy ? cog_count : 1);

                    ASSERT_EQ(mem_fn.calls.get_overread_bytes, 0) << ITER_MSG;
                    ASSERT_EQ(mem_fn.calls.get_scratch_bytes, 0) << ITER_MSG;
                    ASSERT_EQ(mem_fn.calls.memcopy_fn.size(),
                              expected_memcopy_calls)
                        << ITER_MSG;
                    ASSERT_EQ(agg_fn.calls.aggregate_fn.size(),
                              region_pixels * cog_count)
                        << ITER_MSG;
                    ASSERT_EQ(ot_fn.calls.output_transform_fn.size(),
                              region_pixels * cog_count)
                        << ITER_MSG;

                    // iterate through whohe output tensor
                    for (int y_h = 0; y_h < y_height; y_h++) {
                      for (int y_w = 0; y_w < y_width; y_w++) {
                        for (int y_ch = 0; y_ch < y_channels; y_ch++) {
                          // check if the region is in the output space

                          auto in_region = ir.Within(y_h, y_w, y_ch);

                          int expected = in_region ? 1 : 0;
                          int actual = Y[y_h][y_w][y_ch];

                          ASSERT_EQ(expected, actual)
                              << ITER_MSG << " | Output Coords"
                              << ImageVect(y_h, y_w, y_ch);
                        }
                      }
                    }
#undef ITER_MSG
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

}  // namespace nn
