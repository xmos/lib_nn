
#include "Conv2d.hpp"

using namespace nn;

int main(int argc, char **argv)
{

  int x_height = 1;
  int x_width = 1;
  int x_channels = 1;
  int k_height = 1;
  int k_width = 1;
  int k_depth = 1;
  int k_h_dilation = 1;
  int k_v_dilation = 1;
  int k_h_stride = 1;
  int k_v_stride = 1;
  int top_pad = 0;
  int left_pad = 0;
  int right_pad = 0;
  int bottom_pad = 0;

  padding_t padding = {(int16_t)top_pad, (int16_t)left_pad, (int16_t)bottom_pad, (int16_t)right_pad};

  int output_height = CONV2D_OUTPUT_LENGTH(x_height + padding.top + padding.bottom, k_height, k_v_dilation, k_v_stride);
  int output_width = CONV2D_OUTPUT_LENGTH(x_width + padding.left + padding.right, k_width, k_h_dilation, k_h_stride);

  //here output_height + width muct match the allocated memory for y
  ImageGeometry Y(output_height, output_width, k_depth);

  ImageGeometry X(x_height, x_width, x_channels);

  WindowGeometry K(k_height, k_width, k_depth,
                   -padding.top, -padding.left,
                   k_v_stride, k_h_stride, 1,
                   k_v_dilation, k_h_dilation);

  Filter2dGeometry geom(X, Y, K);

  std::vector<int8_t> weights(geom.window.shape.height * geom.window.shape.width * geom.output.depth * geom.input.depth, 0);
  std::vector<int32_t> bias(geom.output.depth);
  std::vector<float> eff_mult(geom.output.depth);
  std::vector<int8_t> input(geom.input.height * geom.input.width * geom.input.depth, 0);
  int input_zero_point = 0;
  int output_zero_point = 0;

  ImToColPadded::Params im_to_col_params(X, K, padding, x_channels, input_zero_point);
  ImToColPadded memcpy(&im_to_col_params);

  int input_bytes = geom.getReceptiveVolumeBytes();
  int scratch_bytes = MatMulInt8::get_scratch_size(input_bytes) + 32;
  std::vector<int8_t> T(scratch_bytes, 0);

  int8_t kernel_pad_val = 0; //This can be anything, 0 in pratice.
  std::array<int, 4> shape = {{k_depth, k_height, k_width, x_channels}};
  Conv2dReorderedWeights rw =
      MatMulInt8::reorder_kernel_weights((int8_t *)weights.data(), shape, 8, kernel_pad_val);

  MatMulInt8::Params p(k_depth, input_bytes, rw.weights.data());
  MatMulInt8 aggregator(&p);

  OutputTransformFnInt8::CanonicalMulAndBias canonical_values =
      OutputTransformFnInt8::canonicalise_mul_and_bias(eff_mult, bias, weights, input_zero_point, output_zero_point, k_depth);

  QuantisationParams qp = OutputTransformFnInt8::quantise_activation(canonical_values.f_multipliers, canonical_values.f_biases,
                                                                     canonical_values.accu_min, canonical_values.accu_max);
  OT_int8::Params ot_params((int32_t)k_depth, &qp.otv, qp.biases.data(),
                            qp.multipliers.data());

  OT_int8 ot(&ot_params);
  auto ir = ImageRegion(0, 0, 0,
                        Y.height,
                        Y.width, Y.depth);

  Filter2D::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

  Conv2dPaddedInDirect conv2d(
      &akp,
      &memcpy,
      &aggregator,
      &ot, &T[0]);

  auto output = std::vector<int8_t>(Y.height * Y.width * Y.depth);

  conv2d.execute(&output[0], &input[0]);

  return 0;
}
