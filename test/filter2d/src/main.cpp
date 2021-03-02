
#include "../src/filt2d/f2d.hpp"

#include <iostream>



const vpu_split_acc32_t biases[1] = {{{0},{0}}};

int8_t kernel_tensor[16][3][3][32];

const nn_acc32_to_int8_params_t ot_params[1] = {{{2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,},{1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,},{0},{0},{0}}};

int8_t img_input[3][3][32];
int8_t img_output[1][1][16];

int8_t patch_mem[3][3][32];


using namespace nn::filt2d;

int main()
{  

  memset(kernel_tensor, 1, sizeof(kernel_tensor));
  memset(img_input, 1, sizeof(img_input));

  auto const filt_geom = geom::Filter2dGeometry<>(
                              geom::ImageGeometry<>(3,3,32),
                              geom::ImageGeometry<>(1,1,16),
                              geom::WindowGeometry<>(3, 3, 32, 0, 0));


  auto filt = Filter2d<int8_t, int8_t, vpu_split_acc32_t, 16, 
                          ValidDeepMemCopyHandler<int8_t>,
                          Conv2dDeepPatchAggregator<>,
                          Int8OutputTransformHandler>(
                              ValidDeepMemCopyHandler<>(&patch_mem[0][0][0], filt_geom),
                              Conv2dDeepPatchAggregator<>(biases, &kernel_tensor[0][0][0][0], filt_geom.window),
                              Int8OutputTransformHandler(ot_params, false),
                              filt_geom);

  filt.bind(&img_input[0][0][0], &img_output[0][0][0]);

  printf("Executing..\n");
  filt.execute(filt_geom.GetFullJob());
  printf("Executed..\n");


  for(int row = 0; row < filt_geom.output.height; row++){
    printf("{ ");
    for(int col = 0; col < filt_geom.output.width; col++){
      printf("{ ");
      for(int chn = 0; chn < filt_geom.output.channels; chn++)
        printf("%d, ", img_output[row][col][chn]);
      printf("}, ");
    }
    printf("},\n");
  }

  printf("Done!\n");

  return 0;
}