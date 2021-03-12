

#include "../src/filt2d/util/conv2d_utils.hpp"
#include "../src/filt2d/f2d.hpp"
#include "gtest/gtest.h"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

#include <memory>
#include <iostream>

using namespace nn::filt2d;

// class Conv2dDeepFilter_Valid_Test : public ::testing::Test {

//   protected:

//     const geom::Filter2dGeometry<int8_t, int8_t> filter_geometry;

//     std::unique_ptr<int8_t[]> kernel_tensor;
//     std::unique_ptr<vpu_split_acc32_t[]> biases;
//     std::unique_ptr<nn_acc32_to_int8_params_t[]> ot_params;
    
//     std::unique_ptr<int8_t[]> patch_mem;


//     Conv2dDeepFilter_Valid_Test(
//       const geom::Filter2dGeometry<int8_t, int8_t> geometry)
//         : filter_geometry(geometry) { 

//       const unsigned kernel_tensor_elms = filter_geometry.window.windowElements() * filter_geometry.output.channels;
//       const unsigned cogs = (filter_geometry.output.channels + 15) >> 4;

//       kernel_tensor = std::unique_ptr<int8_t[]>( new int8_t[kernel_tensor_elms] );
//       biases = std::unique_ptr<vpu_split_acc32_t[]>( new vpu_split_acc32_t[cogs] );
//       ot_params = std::unique_ptr<nn_acc32_to_int8_params_t[]>( new nn_acc32_to_int8_params_t[cogs] );
//       patch_mem = std::unique_ptr<int8_t[]>( new int8_t[ filter_geometry.window.windowElements() ]);

//     }

//     void SetUp() override {



//     }

//     void TearDown() override {

//     }



// };


namespace nn::test::filt2d::op {





struct ExtraRefParams {
  struct { int32_t zero_point; } input, filter;
  struct {
    int32_t zero_point;
    int32_t* multiplier;
    int32_t* shift;
    struct { int32_t min, max; } activation;
  } output;
};

std::unique_ptr<int8_t[]> Conv2dDeepReference(
    geom::Filter2dGeometry<int8_t,int8_t>& geom,
    const int8_t* input_data, const int8_t* filter_data,
    const int32_t* bias_data, const ExtraRefParams& ref_params)
{
  tflite::ConvParams op_params;

  auto pad_initial = geom.InitialPadding();
  auto pad_final   = geom.FinalPadding();
  op_params.padding_type = tflite::PaddingType::kSame;
  op_params.padding_values.height = pad_initial.top;
  op_params.padding_values.width = pad_initial.left;
  op_params.padding_values.height_offset = pad_final.bottom;
  op_params.padding_values.width_offset = pad_final.right;

  op_params.stride_height = geom.window.stride.row;
  op_params.stride_width  = geom.window.stride.col;
  op_params.dilation_height_factor = geom.window.dilation.row;
  op_params.dilation_width_factor  = geom.window.dilation.col;

  op_params.input_offset = -ref_params.input.zero_point;
  op_params.output_offset = ref_params.output.zero_point;
  op_params.weights_offset = -ref_params.filter.zero_point;

  // When per-channel quantization is used, these values are ignored. Needs the arrays of multipliers and shifts
  // op_params.output_multiplier = ref_params.output.multiplier;
  // op_params.output_shift = ref_params.output.shift;

  op_params.quantized_activation_min = ref_params.output.activation.min;
  op_params.quantized_activation_max = ref_params.output.activation.max;

  struct {
    tflite::RuntimeShape input, output, bias, filter, im2col;
  } shape = {
      .input = {  1, (int) geom.input.height,  (int) geom.input.width,  (int) geom.input.depth },
      .output = { 1, (int) geom.output.height, (int) geom.output.width, (int) geom.output.depth },
      .bias = { (int) geom.output.depth },
      .filter = { (int) geom.output.depth, (int) geom.window.shape.height, 
                  (int) geom.window.shape.width, (int) geom.input.depth },
      .im2col = { (int) geom.window.windowElements() },
  };

  std::unique_ptr<int8_t[]> output_data( new int8_t[geom.output.imageElements()] );

  tflite::reference_integer_ops::ConvPerChannel(op_params, 
                              ref_params.output.multiplier, ref_params.output.shift,
                              shape.input, input_data,
                              shape.filter, filter_data,
                              shape.bias, bias_data,
                              shape.output, output_data.get());

  return output_data;
}




TEST(JustMeTesting, ShutUp){

  auto filt_geom = geom::Filter2dGeometry<int8_t,int8_t>(
                      geom::ImageGeometry<int8_t>(1, 1, 32),
                      geom::ImageGeometry<int8_t>(1, 1, 16),
                      geom::WindowGeometry<int8_t>(1, 1, 32));

  int8_t input[1][1][32];
  int8_t kernel[16][1][1][32];

  vpu_split_acc32_t biases[1];
  int32_t ref_biases[16];

  float effective_output_multiplier[16];

  int32_t ref_output_multiplier[16];
  int32_t ref_output_shift[16];

  memset(input,  1, sizeof(input ));
  memset(kernel, 1, sizeof(kernel));
  memset(biases, 0, sizeof(biases));
  memset(ref_biases, 0, sizeof(ref_biases));

  for(int i = 0; i < 16; ++i){
    effective_output_multiplier[i] = 1.0f;
    nn::filt2d::conv2d::util::TfLiteConverter::QuantizeEffectiveOutputMultiplier(
                                                ref_output_multiplier[i], 
                                                ref_output_shift[i],
                                                effective_output_multiplier[i] );
  }

  ExtraRefParams ref_params;
  ref_params.input.zero_point = 0;
  ref_params.filter.zero_point = 0;
  ref_params.output.zero_point = 0;
  ref_params.output.multiplier = ref_output_multiplier;
  ref_params.output.shift = ref_output_shift;
  ref_params.output.activation.min = std::numeric_limits<int8_t>::min();
  ref_params.output.activation.max = std::numeric_limits<int8_t>::max();

  auto output = Conv2dDeepReference(filt_geom, &input[0][0][0], &kernel[0][0][0][0], ref_biases, ref_params);

  std::cout << "Ref output: [ ";
  for(int i = 0; i < 16; i++)
    std::cout << static_cast<int>(output.get()[i]) << ", ";
  std::cout << "]" << std::endl;

  // using nn::filt2d::conv2d::util::TfLiteConverter;
  std::vector<vpu_split_acc32_t> xcore_biases = 
      nn::filt2d::conv2d::util::TfLiteConverter::ConvertBiases(
                                                    filt_geom, &kernel[0][0][0][0], 
                                                    ref_biases, ref_params.input.zero_point, false);

  std::vector<nn_acc32_to_int8_params_t> xcore_output_params = 
      nn::filt2d::conv2d::util::TfLiteConverter::ConvertOutputParams(
                                                    filt_geom, effective_output_multiplier,
                                                    ref_params.output.zero_point);


  std::vector<int8_t> xcore_output(filt_geom.output.imageElements());
  std::vector<int8_t> patch_mem(filt_geom.window.windowBytes());

  auto filter = nn::filt2d::op::Conv2dDeepFilter_Valid(
                      &input[0][0][0], 
                      &xcore_output[0],
                      filt_geom, 
                      &xcore_biases[0], 
                      &kernel[0][0][0][0],
                      &xcore_output_params[0], false);


  filter.execute(&patch_mem[0]);
  // filter.spawnJob(ImageRegion(0,0,0,1,1,16), &patch_mem[0]).execute();

  
  std::cout << "xCore output: [ ";
  for(int i = 0; i < 16; i++)
    std::cout << static_cast<int>(xcore_output[i]) << ", ";
  std::cout << "]" << std::endl;
}



class RectRange {

  public:

    struct Dim { int start, end, step; };

    class iterator: public std::iterator<std::input_iterator_tag, std::vector<int>> {

      private:

        RectRange& parent;
        std::vector<int> val;

      public:

        explicit iterator(RectRange& parent, std::vector<int> val)
          : parent(parent), val(val) {}

        iterator& operator++(){
          for(int i = val.size()-1; i >= 0; --i){
            auto dim = parent.getDim(i);
            
            val[i] += dim.step;

            if(val[i] < dim.end){
              for(int j = i+1; j < val.size(); ++j)
                val[j] = parent.getDim(j).start; // I know this isn't great, but I think it should work..
              break;
            } else {
              val[i] = dim.end;
            }
          }

          return *this;
        }

        iterator operator++(int){ iterator retval = *this; ++(*this); return retval; }
        bool operator==(iterator other) const { return this->val == other.val; }
        bool operator!=(iterator other) const { return this->val != other.val; }
        reference operator*() { return this->val; }

    };

  private:

    std::vector<Dim> dims;

  public:

    RectRange(std::initializer_list<Dim> il)
      : dims(il) {}

    const Dim& getDim(int i){ return this->dims[i]; }

    iterator begin() {
      auto v = std::vector<int>(dims.size());
      for(int i = 0; i < dims.size(); ++i)
        v[i] = dims[i].start;
      return iterator(*this, v);
    }

    iterator end() {
      auto v = std::vector<int>(dims.size());
      for(int i = 0; i < dims.size(); ++i)
        v[i] = dims[i].end;
      return iterator(*this, v);
    }

};



TEST(JustTryingSomething,RectRangeStuff){
  
  RectRange rr = { 
  //  start,   end,   step 
    {     0,     5,      1 }, // Iterates slowest 
    {     0,     3,      1 }, 
    {     0,     4,      2 }  // Iterates fastest
  };

  for(auto v: rr){

    std::cout << "iter: " << "(";
    for(int i = 0; i < v.size(); i++)
      std::cout << v[i] << ", ";
    std::cout << ")" << std::endl;

  }
}


#define X_CHAN    8
#define Y_CHAN    16
#define X_SIZE    5
#define K_SIZE    3
#define Y_SIZE    3

#define COGS    (((Y_CHAN)+15) >> 4)

const vpu_split_acc32_t biases[COGS] = {{{0}}};

int8_t kernel_tensor[Y_CHAN][K_SIZE][K_SIZE][X_CHAN];

nn_acc32_to_int8_params_t ot_params[COGS];

int8_t img_input[X_SIZE][X_SIZE][X_CHAN];
int8_t img_output[Y_SIZE][Y_SIZE][Y_CHAN];

int8_t patch_mem[K_SIZE*K_SIZE*X_CHAN + 32];


using namespace nn::filt2d;

TEST(Conv2dDeepFilter_Valid, basicCase0)
{  
  memset(patch_mem, 0, sizeof(patch_mem));
  memset(kernel_tensor, 1, sizeof(kernel_tensor));
  memset(img_input, 1, sizeof(img_input));
  memset(ot_params, 0, sizeof(ot_params));

  for(int i = 0; i < Y_CHAN; i++){
    ot_params[ i>>4 ].shift1[ i%16 ] = 3;
    ot_params[ i>>4 ].scale [ i%16 ] = 1;
  } 

  auto const filt_geom = geom::Filter2dGeometry<>(
                              geom::ImageGeometry<>(X_SIZE,X_SIZE,X_CHAN),
                              geom::ImageGeometry<>(Y_SIZE,Y_SIZE,Y_CHAN),
                              geom::WindowGeometry<>(K_SIZE, K_SIZE, X_CHAN, 0, 0));


  auto filter = nn::filt2d::op::Conv2dDeepFilter_Valid(
                      &img_input[0][0][0], 
                      &img_output[0][0][0],
                      filt_geom, 
                      biases, 
                      &kernel_tensor[0][0][0][0],
                      ot_params, 
                      false);

  printf("Executing..\n");
  // filter.execute(patch_mem);
  filter.spawnJob(ImageRegion(0,0,0,1,Y_SIZE,Y_CHAN), patch_mem).execute();
  printf("Executed..\n");



  for(int row = 0; row < filt_geom.output.height; row++){
    printf("{ ");
    for(int col = 0; col < filt_geom.output.width; col++){
      printf("{ ");
      for(int chn = 0; chn < filt_geom.output.depth; chn++)
        printf("%d, ", img_output[row][col][chn]);
      printf("}, ");
    }
    printf("},\n");
  }

  printf("Done!\n");

}


}