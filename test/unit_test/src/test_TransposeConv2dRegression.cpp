#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "AbstractKernel.hpp"
#include "AggregateFn.hpp"
#include "MemCpyFn.hpp"
#include "OutputTransformFn.hpp"
#include "Rand.hpp"
#include "RefOps.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "geom/util.hpp"
#include "nn_types.h"
#include "TransposeConv.h"

extern "C" {
#include "tst_common.h"
#include "unity.h"
}

using namespace nn;
using namespace nn::test;

static auto rng = Rand(69);

const int max_k_channels = 32;
const int itt_count = 1;

static unsigned deref(const unsigned r, const unsigned len, const unsigned idx){
     return len*r + idx;
}
static unsigned deref4d(const unsigned dim0, const unsigned dim1, 
            const unsigned dim2, const unsigned dim3, 
            const unsigned idx0, const unsigned idx1, 
            const unsigned idx2, const unsigned idx3){
    return deref(deref(deref(deref(0, dim0, idx0), dim1, idx1), dim2, idx2), dim3, idx3);
}

struct KernelStimulus {
  std::vector<int8_t> weights;
  std::vector<int32_t> bias;
  std::vector<float> eff_mult;
  std::vector<int8_t> input;
  int input_zero_point;
  int output_zero_point;

  KernelStimulus(Filter2dGeometry &geom)
      : weights(geom.window.shape.height * geom.window.shape.width *
                    geom.output.depth * geom.input.depth,
                0),
        bias(geom.output.depth),
        eff_mult(geom.output.depth),
        input(geom.input.height * geom.input.width * geom.input.depth, 0),
        input_zero_point(0),
        output_zero_point(0){};
};

KernelStimulus create_simple_stimulus2(Filter2dGeometry &geom) {
  KernelStimulus ks(geom);

  for (int idx = 0; idx < ks.weights.size(); ++idx)
    ks.weights[idx] = rng.rand<int8_t>();

  for (int idx = 0; idx < ks.input.size(); ++idx)
    ks.input[idx] = rng.rand<int8_t>();

  ks.input_zero_point = 0; //rng.rand<int8_t>() % 4;
  ks.output_zero_point = 0; //rng.rand<int8_t>() % 32;

  for (int ch = 0; ch < geom.output.depth; ch++) {
    ks.eff_mult[ch] = 1.0;//rng.rand<float>(1.0 / (256 * 256), 1.0 / 256);
    ks.bias[ch] = 0.0;//rng.rand<int8_t>() % 512;
  }
  float eff_mult_scalar =0.0025;
  for (int ch = 0; ch < geom.output.depth; ch++) {
    ks.eff_mult[ch] *= eff_mult_scalar;
  }
  return ks;
}

void test_TransposeConv2dPaddedIndirectRegression() {

  int tid = 0;
  for (int x_height = 1; x_height <= 6; ++x_height) {
    for (int x_width = 1; x_width <= 6; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {

        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {

            for (int k_depth = 4; k_depth <= max_k_channels; k_depth += 4) {
              for (int k_h_dilation = 1; k_h_dilation <= 1; ++k_h_dilation) { //dont do this yet
                for (int k_v_dilation = 1; k_v_dilation <= 1; ++k_v_dilation) {

                  for (int k_h_stride = 1; k_h_stride <= k_width; ++k_h_stride) { //the strides cannot be greater than the kernel size
                    for (int k_v_stride = 1; k_v_stride <= k_height; ++k_v_stride) {

                      for (int top_pad = 0; top_pad <= 0; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 0; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 0; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 0;
                                 ++bottom_pad) {
                              for (int itt = 0; itt < itt_count; ++itt) {

                                padding_t padding = {
                                    (int16_t)top_pad, (int16_t)left_pad,
                                    (int16_t)bottom_pad, (int16_t)right_pad};

                                int output_height = TRANSPOSE_CONV2D_OUTPUT_LENGTH(
                                    x_height, k_height, k_v_stride, 0, right_pad + left_pad);
                                int output_width = TRANSPOSE_CONV2D_OUTPUT_LENGTH(
                                    x_width, k_width, k_h_stride, 0, top_pad + bottom_pad);



                                if (output_height <= 0 || output_width <= 0)
                                  continue;

                                tid++;  //this just count the number of tests

                                int test_seed = rng.getSeed();

                                // here output_height + width muct match the
                                // allocated memory for y
                                ImageGeometry Y(output_height, output_width,
                                                k_depth);

                                ImageGeometry X(x_height, x_width, x_channels);

                                WindowGeometry K(k_height, k_width, k_depth,
                                                 -padding.top, -padding.left,
                                                 k_v_stride, k_h_stride, 1,
                                                 k_v_dilation, k_h_dilation);

                                Filter2dGeometry geom(X, Y, K);

                                KernelStimulus ks =
                                    create_simple_stimulus2(geom);
                                auto &weights = ks.weights;
                                auto &bias = ks.bias;
                                auto &eff_mult = ks.eff_mult;

                                int8_t kernel_pad_val =
                                    rng.rand<int8_t>();  // This can be anything, 0 in pratice.

                                //This is the shape of the conv2d transpose kernel
                                std::array<int, 4> original_kernel_shape = {
                                    {k_depth, k_height, k_width, x_channels}};

                                std::vector<ConvParams> convParams =  transpose_conv_reorder_kernel_weights(
                                     (int8_t *)weights.data(), original_kernel_shape, k_v_stride, k_h_stride);

                                //this works out the amount of padding that an operator will need to insert before the convs begin.
                                int vertical_padding = 0;
                                int horizontal_padding = 0;
                                for(ConvParams c : convParams) {
                                    std::array<int, 4>  sub_kernel_shape = c.kernelShape;
                                    vertical_padding = std::max(vertical_padding, sub_kernel_shape[1]-1);
                                    horizontal_padding = std::max(horizontal_padding, sub_kernel_shape[2]-1);
                                }
                                // std::cout << " x_height:" << x_height
                                //         << " x_width:" << x_width
                                //         << " x_channels:" << x_channels
                                //         << " k_height:" << k_height
                                //         << " k_width:" << k_width
                                //         << " k_depth:" << k_depth
                                //         << " k_h_stride:" << k_h_stride
                                //         << " k_v_stride:" << k_v_stride
                                //         // << " top_pad:" << top_pad
                                //         // << " left_pad:" << left_pad
                                //         // << " right_pad:" << right_pad
                                //         // << " bottom_pad:" << bottom_pad
                                //         << " output_height:" << output_height
                                //         << " output_width:" << output_width
                                //         << " vertical_padding:" << vertical_padding
                                //         << " horizontal_padding:" << horizontal_padding
                                //         << " conv count:" << convParams.size()
                                //         << std::endl;

                                int8_t padded_input[x_height + 2*vertical_padding][x_width+2*horizontal_padding][x_channels];
                                std::memset(padded_input, 0, sizeof(padded_input));
                                int8_t * input = (int8_t*)padded_input;

                                ImageGeometry X_padded(x_height + 2*vertical_padding, x_width+2*horizontal_padding, x_channels); //this is X padded to account for the fixed padding we are going to add

                                for(int h=0;h<x_height;h++){
                                  for(int w=0;w<x_width;w++){
                                    for(int c=0;c<x_channels;c++){
                                      int idx = deref4d(1, x_height, x_width, x_channels, 0, h, w, c);
                                      padded_input[vertical_padding + h][horizontal_padding+w][c] = ks.input[idx];
                                    }
                                  }
                                }

                                auto expected =
                                    nn::test::ops::ref::TransposeConv2dDenseReference(
                                        geom, ks.input.data(), weights.data(),
                                        bias.data(), eff_mult.data(),
                                        ks.input_zero_point,
                                        ks.output_zero_point);

                                //Allocate and zero the output tensor
                                alignas(4) int8_t output[Y.height* Y.width* Y.depth];
                                std::memset(output, 0, sizeof(output));

                                for(ConvParams c : convParams) {
                                    //apply each convolution, writing the result to the correct location in the output
                                    int subH = c.subH;
                                    int subW = c.subW;
                                    std::array<int, 4>  sub_kernel_shape = c.kernelShape;
  
                                    Conv2dReorderedWeights rw =
                                      MatMulInt8::reorder_kernel_weights(
                                          (int8_t *)c.weights.data(), sub_kernel_shape, 8,
                                          kernel_pad_val);

                                    int input_bytes = sub_kernel_shape[1] *
                                                      sub_kernel_shape[2] *
                                                      sub_kernel_shape[3];
                                    int scratch_bytes =
                                        MatMulInt8::get_scratch_mem_bytes(
                                            input_bytes) +
                                        32;
                                    std::vector<int8_t> T(scratch_bytes, 0);

                                    MatMulInt8 aggregator(k_depth, input_bytes);

                                    assert(eff_mult.size() > 0);

                                    MulsAndBias mul_and_biases =
                                        OutputTransformFnInt8::
                                            canonicalise_mul_and_bias(
                                                eff_mult, bias, c.weights,
                                                ks.input_zero_point,
                                                ks.output_zero_point, k_depth);

                                    auto quantizer =
                                        OutputTransformFnInt8_Group::Quantizer();
                                    OutputTransformFnInt8_Group::QuantisationParams
                                        qp = quantizer.quantise_activation(
                                            mul_and_biases, false);

                                    assert(qp.multipliers.size() > 0);
                                    assert(qp.biases.size() > 0);

                                    auto serialised_offsets_multipliers_and_biases =
                                        OutputTransformFn::serialise_memory(
                                            qp.multipliers, qp.biases);

                                    // pad qp.multipliers_and_biases to a multiple
                                    // of VPU_INT16_EPV this is to work around array
                                    // over reads
                                    int16_t pad_val =
                                        rng.rand<int16_t>();  // this is arbitrary
                                    OutputTransformFn::pad_final_access(
                                        serialised_offsets_multipliers_and_biases,
                                        VPU_INT16_EPV, pad_val);

                                    OT_int8 ot((int32_t)k_depth,
                                                              qp.initial_shr,
                                                              qp.final_shr);
                                    assert(qp.multipliers.size() > 0);
                                    assert(qp.biases.size() > 0);

                                    int this_kernels_vertical_padding = sub_kernel_shape[1] - 1;
                                    int this_kernels_horizontal_padding = sub_kernel_shape[2] - 1;

                                    auto ir = ImageRegion(0, 0, 0, Y.height,
                                                          Y.width, Y.depth);
                                    AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD ,
                                                     subH, subW,
                                                     k_v_stride, k_h_stride,
                                                     sub_kernel_shape[1],
                                                     sub_kernel_shape[2]);
                                    
                                    abstract_kernel_params_t a = akp.getParams();
                                    
                                    WindowGeometry K_boggled(sub_kernel_shape[1], sub_kernel_shape[2], sub_kernel_shape[3],
                                                 -padding.top, -padding.left,
                                                 1, 1, 1,
                                                 k_v_dilation, k_h_dilation);

                                    ImToColPadded memcpy(
                                      X_padded, K_boggled, padding, x_channels,
                                      ks.input_zero_point);

                                    memcpyfn_imtocol_padded_params_t m = memcpy.getParams();
                                    mat_mul_generic_params_t agg = aggregator.getParams();
                                    otfn_int8_params_t o = ot.getParams();
                                    conv_params_t params;

                                    params.memcopy_fn = (MemFnType)memcpyfn_imtocol_padded;
                                    params.aggregate_fn = (AggFnType)mat_mul_generic_int8;
                                    params.output_transform_fn = (OtFnType)otfn_int8;
                                    params.mem_p = &m;
                                    params.agg_p = &agg;
                                    params.ot_p = &o;

                                    int8_t * input_start_addr = &input[0]+x_channels*(horizontal_padding - this_kernels_horizontal_padding) + 
                                                              (x_channels*(vertical_padding - this_kernels_vertical_padding) *  (x_width+2*horizontal_padding));

                                    nn::execute(&output[0], input_start_addr, &params,
                                                &a, rw.weights.data(), serialised_offsets_multipliers_and_biases.data(), /*isConv=*/true, &T[0]);

                                }

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    int pixel_correct = 1;
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                        int diff =  (int)expected[idx] - (int)output[idx];
                                        if(diff < 0) diff = -diff;
                                        pixel_correct &= diff <= 1;
                                    }

                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      TEST_ASSERT_INT32_WITHIN(
                                          1, (int)expected[idx],
                                          (int)output[idx]);
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
                }
              }
            }
          }
        }
      }
    }
  }
  printf("done %d\n", tid);
}

extern "C" void test_transposeconv2d_regression();
void test_transposeconv2d_regression() {
  UNITY_SET_FILE();
  RUN_TEST(test_TransposeConv2dPaddedIndirectRegression);
}
