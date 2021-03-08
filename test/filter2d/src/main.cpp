
#include "../src/filt2d/f2d.hpp"
#include "../src/filt2d/Conv2dDeepFilter.hpp"

#include <iostream>
#include <cstring>

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

int main()
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


  auto filter = Conv2dDeepFilter(&img_input[0][0][0], &img_output[0][0][0],
                                 filt_geom, biases, &kernel_tensor[0][0][0][0],
                                 ot_params, false);

  printf("Executing..\n");
  // filter.execute(patch_mem);
  filter.spawnJob(ImageRegion(0,0,0,1,Y_SIZE,Y_CHAN), patch_mem).execute();
  printf("Executed..\n");

  // auto filt = Filter2d<int8_t, int8_t, vpu_split_acc32_t, 16, 
  //                         ValidDeepMemCopyHandler<int8_t>,
  //                         Conv2dDeepPatchAggregator<>,
  //                         Int8OutputTransformHandler>(
  //                             ValidDeepMemCopyHandler<>(&patch_mem[0][0][0], filt_geom),
  //                             Conv2dDeepPatchAggregator<>(biases, &kernel_tensor[0][0][0][0], filt_geom.window),
  //                             Int8OutputTransformHandler(ot_params, false),
  //                             filt_geom);

  // filt.bind(&img_input[0][0][0], &img_output[0][0][0]);

  // printf("Executing..\n");
  // filt.execute(filt_geom.GetFullJob());
  // printf("Executed..\n");


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
