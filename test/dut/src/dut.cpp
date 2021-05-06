
#include "Conv2d.hpp"

using namespace nn;

#include <stdio.h>
#include <array>
#include "Rand.hpp"

namespace nn
{

    static auto rng = test::Rand(42);
    void foo()
    {
        printf("hello\n");
        const int vpu_ring_buffer_length = 16;

        int k_h_stride = 1;
        int k_v_stride = 1;

        //TODO replace 16 and 32
        for (int x_height = 1; x_height <= 4; ++x_height)
        {
            for (int x_width = 1; x_width <= 3; ++x_width)
            {
                for (int x_channels = 32; x_channels <= 32 * 3; x_channels += 32)
                {
                    for (int k_height = 1; k_height <= x_height; ++k_height)
                    {
                        for (int k_width = 1; k_width <= x_width; ++k_width)
                        {
                            for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride)
                            {
                                for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride)
                                {
                                    for (int k_h_dilation = 1; k_h_dilation <= 4; ++k_h_dilation)
                                    {
                                        for (int k_v_dilation = 1; k_v_dilation <= 4; ++k_v_dilation)
                                        {
                                            for (int output_channels = 16; output_channels <= 16 * 3; output_channels += 16)
                                            {
                                                for (int input_ch_per_output = x_channels; input_ch_per_output <= x_channels; input_ch_per_output += 32)
                                                {

                                                    int output_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, k_v_dilation, k_v_stride);
                                                    int output_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, k_h_dilation, k_h_stride);

                                                    if (output_height <= 0 || output_width <= 0)
                                                        continue;
                                                    // std::cout << "x_height: " << x_height
                                                    //           << " x_width: " << x_width
                                                    //           << " x_channels: " << x_channels
                                                    //           << " k_height: " << k_height
                                                    //           << " k_width: " << k_width
                                                    //           << " k_h_dilation: " << k_h_dilation
                                                    //           << " k_v_dilation: " << k_v_dilation
                                                    //           << " output_channels: " << output_channels
                                                    //           << " input_ch_per_output: " << input_ch_per_output
                                                    //           << std::endl;

                                                    ImageGeometry X(x_height, x_width, x_channels);
                                                    WindowGeometry K(k_height, k_width, 0,
                                                                     0, 0,
                                                                     k_v_stride, k_h_stride, 0, k_v_dilation, k_h_dilation);

                                                    std::array<int, 4> shape = {{output_channels, k_height, k_width, x_channels}};
                                                    int8_t raw_weights[output_channels][k_height][k_width][x_channels];

                                                    for (int j = 0; j < sizeof raw_weights; ++j)
                                                        ((int8_t *)raw_weights)[j] = rng.rand<int8_t>();

                                                    int8_t X_mem[x_height][x_width][x_channels];

                                                    for (int j = 0; j < sizeof X_mem; ++j)
                                                        ((int8_t *)X_mem)[j] = rng.rand<int8_t>();

                                                    int8_t pad_val = rng.rand<int8_t>(); //this should be unused in this case

                                                    Conv2dReorderedWeights rw =
                                                        MatMulInt8::reorder_kernel_weights((int8_t *)raw_weights, shape, 8, pad_val);

                                                    MatMulDirectFn::Params p(X, K, input_ch_per_output, rw.weights.data());
                                                    MatMulDirectFn mmd(&p);

                                                    int ocg_count = (output_channels + vpu_ring_buffer_length - 1) / vpu_ring_buffer_length;
                                                    //printf("X_mem: %p\n", X_mem);
                                                    for (int ocg = 0; ocg < ocg_count; ++ocg)
                                                    {

                                                        vpu_ring_buffer_t A;
                                                        mmd.aggregate_fn(&A, (int8_t *)X_mem, ocg);

                                                        int chs_in_group = std::min(output_channels - vpu_ring_buffer_length * ocg, vpu_ring_buffer_length);

                                                        for (int output_chan = 0; output_chan < chs_in_group; ++output_chan)
                                                        {

                                                            int actual_output_channel = output_chan + ocg * vpu_ring_buffer_length;

                                                            int expected_sum = 0;

                                                            for (int h = 0; h < k_height; ++h)
                                                            {
                                                                for (int w = 0; w < k_width; ++w)
                                                                {
                                                                    for (int c = 0; c < input_ch_per_output; ++c)
                                                                    {
                                                                        int x = (int)X_mem[k_v_dilation * h][k_h_dilation * w][c];
                                                                        int t = raw_weights[actual_output_channel][h][w][c];
                                                                        expected_sum += x * t;
                                                                    }
                                                                }
                                                            }

                                                            int32_t v;
                                                            ((int16_t *)&v)[0] = A.vD[output_chan];
                                                            ((int16_t *)&v)[1] = A.vR[output_chan];
                                                            //std::cout << v << " " << expected_sum << std::endl;
                                                            assert(v == expected_sum);
                                                            // EXPECT_EQ(v, expected_sum);
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
        printf("done\n");
    }

    int bar(int argc, char **argv)
    {

        printf("hello\n");
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

        for (auto o : output)
        {
            std::cout << (int)o << std::endl;
        }

        return 0;
    }
}

int main(int argc, char **argv)
{
    foo();
}