#include <cstring>
#include <vector>

#include "AbstractKernel.hpp"
#include "Rand.hpp"
#include "gtest/gtest.h"
#include "xs3_vpu.h"

namespace nn {

class MockAbstractKernel : public AbstractKernel {
 public:
  int32_t output_slice_channel_count;

  MockAbstractKernel(AbstractKernel::Params *kparams,
                     int32_t output_slice_channel_count)
      : AbstractKernel(kparams),
        output_slice_channel_count(output_slice_channel_count) {}

  void calc_output_pixel_slice(int8_t *output_image, int8_t *input_image,
                               int32_t output_row, int32_t output_col) {
    std::memset(output_image, 1, sizeof(int8_t) * output_slice_channel_count);
  }
};

class Test_AbstractKernel : public ::testing::Test {};

TEST_F(Test_AbstractKernel, BasicTest) {
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

                    AbstractKernel::Params akp(ip, ir, cog_size);

                    MockAbstractKernel f(&akp,
                                         r_channels_end - r_channels_start);

                    int8_t Y[y_height][y_width][y_channels];

                    std::memset(Y, 0, sizeof(Y));

                    f.execute((int8_t *)Y, nullptr);

#define ITER_MSG "Y: " << ip << " | Output Region: " << ir

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
