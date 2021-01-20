#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "tst_common.h"
#include "unity.h"

#include "helpers.h"

#define X_REF_OVERREAD_WORDS (7)
#define DATA_SCRATCH_OVERREADWRITE_WORDS (8)

static const char undef_sentinal = 0x55;


/*
X_ref and K_ref must be initialised before running this.
This function test whole images, i.e. it wont work on a sub image.
*/
static void run_int8_config(int8_t* Y_p, int8_t* Y_ref_p, bnn_b32_t* X_ref,
               bnn_b32_t* K_p, bnn_b32_t* K_ref_p, 

               float* post_activation_multiplier,
               float* post_activation_bias, 

               int16_t * post_activation_multiplier_q,
               int16_t * post_activation_bias_q, 

               int16_t * quantised_accu_modifier,

               int * chan_overlaps,
               
               unsigned x_height, unsigned x_width,
               unsigned k_height, unsigned k_width, unsigned chans_in,
               unsigned chans_out, unsigned h_stride, unsigned v_stride, int seed,

               int32_t larq_clamp_min, 
               int32_t larq_clamp_max, 

               void (*test_fn)()) {
                  
  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u seed:%d\n", 
  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in, chans_out, seed);

  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);

  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned receptive_volume = k_width * k_height * chans_in;

  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    Y_ref_p[e]=0;
  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    Y_p[e]=0;

  nn_image_params_t x;
  x.height = x_height;
  x.width = x_width;
  x.channels = chans_in;
  nn_image_params_t y;
  y.height = y_height;
  y.width = y_width;
  y.channels = chans_out;
  nn_window_params_t k;
  k.shape.height = k_height;
  k.shape.width = k_width;
  k.stride.horizontal = h_stride;
  k.stride.vertical = v_stride;
  k.dilation.horizontal = 1;
  k.dilation.vertical = 1;

  larq_ref_bconv2d_int8_out(&x, &y, &k, (int32_t*)X_ref, (int32_t*)K_ref_p,
                   (int8_t*)Y_ref_p, post_activation_multiplier, post_activation_bias, larq_clamp_min, larq_clamp_max);

  bnn_reorder_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
                            chans_out, chan_overlaps);

  int16_t bias_multipler;
  int accu_shr, final_shr;

  int16_t low_clamp_offset;
  int16_t high_clamp_offset;

  bnn_quantise_activation(
      post_activation_multiplier_q,
      post_activation_bias_q,

      post_activation_multiplier,
      post_activation_bias, 

      chans_out,

      larq_clamp_min, 
      larq_clamp_max,

      quantised_accu_modifier,
      &low_clamp_offset,
      &high_clamp_offset,

      &accu_shr, &bias_multipler, &final_shr, receptive_volume, chan_overlaps
  );

  // for (unsigned e=0;e<chans_out;++e){
  //   printf("%d %d\n", e, chan_overlaps[e]);
  // }

  test_fn((int8_t*)Y_p, (const bnn_b32_t*)X_ref,
    (const bnn_b32_t*)K_p, post_activation_multiplier_q, 
    post_activation_bias_q, 
    quantised_accu_modifier, low_clamp_offset, high_clamp_offset,
    accu_shr, bias_multipler, final_shr,
    &x, &y, &k);
    
  // for (unsigned e=0;e<y_height * y_width * chans_out;++e){
  //   printf("%d %d\n", Y_ref_p[e], Y_p[e]);
  // }
  for (unsigned e=0;e<y_height * y_width * chans_out;++e)
    TEST_ASSERT_INT8_WITHIN(1, Y_ref_p[e], Y_p[e]);
  // exit(10);
  //FIXME - why wont this link? The above is a workaround
  // TEST_ASSERT_INT8_ARRAY_WITHIN(1, Y_ref_p, Y_p, y_height * y_width * chans_out);
}

void impl_bconv2d_int8_pseudo_random(
  const unsigned min_k_height, const unsigned max_k_height, 
  const unsigned min_k_width, const unsigned max_k_width,  
  
  const unsigned min_chans_in, const unsigned max_chans_in,    
  const unsigned min_chans_out, const unsigned max_chans_out,  

  const unsigned chans_in_inc, const unsigned chans_out_inc,

  const unsigned min_v_stride, const unsigned max_v_stride, 
  const unsigned min_h_stride, const unsigned max_h_stride,
  void (* valid_impl)()) {

  for (unsigned h_stride = min_h_stride; h_stride <= max_h_stride; ++h_stride) {
    for (unsigned v_stride = min_v_stride; v_stride <= max_v_stride;
         ++v_stride) {
      for (unsigned k_height = min_k_height; k_height <= max_k_height;
           ++k_height) {
        unsigned max_x_height = k_height;
        for (unsigned k_width = min_k_width; k_width <= max_k_width;
             ++k_width) {
          unsigned max_x_width = k_width;

          for (unsigned x_height = k_height; x_height <= max_x_height;
               ++x_height) {
            for (unsigned x_width = k_width; x_width <= max_x_width;
                 ++x_width) {
              unsigned y_height =  CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
              unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);
              
              for (unsigned chans_in = min_chans_in; chans_in <= max_chans_in;
                   chans_in += chans_in_inc) {
                for (unsigned chans_out = min_chans_out;
                     chans_out <= max_chans_out; chans_out += chans_out_inc) {

                  unsigned chan_words_in = chans_in/32;

                  size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out*k_height*k_width*chan_words_in);
                  bnn_b32_t * K_ref = (bnn_b32_t *) malloc(K_ref_bytes);

                  int32_t over_bytes = compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);

                  bnn_b32_t * K     = (bnn_b32_t *) malloc(K_ref_bytes + over_bytes);

                  size_t X_ref_bytes = sizeof(bnn_b32_t)*(x_height*x_width*chan_words_in+X_REF_OVERREAD_WORDS);
                  bnn_b32_t * X_ref =(bnn_b32_t *)malloc(X_ref_bytes);
                  int16_t *post_activation_multiplier_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
                  int16_t *post_activation_bias_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
                  
                  int16_t *quantised_accu_modifier = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          
                  float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
                  float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
                  int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

                  int8_t * Y     = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);
                  int8_t * Y_ref = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);
      
                  assert(X_ref);
                  assert(Y);
                  assert(Y_ref);
                  assert(post_activation_multiplier_q);
                  assert(post_activation_bias_q);
                  assert(quantised_accu_modifier);
                  assert(K);
                  assert(K_ref);

                  assert(post_activation_multiplier);
                  assert(post_activation_bias);
                  assert(chan_overlaps);
                  
                  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u\n", 
                  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in, chans_out);

                    for(unsigned c=0;c<1<<1;c++){
                      int seed = c;

                      for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
                        ((int*)X_ref)[b] = pseudo_rand(&seed);

                      
                      for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
                        ((int*)K_ref)[b] = pseudo_rand(&seed);
                      
                      unsigned receptive_volume = k_width * k_height * chans_in;
                      pick_post_activation_params(post_activation_multiplier, post_activation_bias, chans_out, receptive_volume, &seed);

                      for (unsigned c=0;c<1024;c++){
                        int32_t larq_clamp_min = pseudo_rand(&seed) % (2*receptive_volume);
                        int32_t larq_clamp_max = larq_clamp_min + pseudo_rand(&seed) % (2*receptive_volume);

                        run_int8_config(
                            (int8_t*)Y, (int8_t*)Y_ref, (bnn_b32_t*)X_ref,
                            (bnn_b32_t*)K, (bnn_b32_t*)K_ref,
                            (float*)post_activation_multiplier,
                            (float*)post_activation_bias, 
                            (int16_t*)post_activation_multiplier_q,
                            (int16_t*)post_activation_bias_q,  
                            (int16_t*)quantised_accu_modifier,
                            (int*) chan_overlaps,
                            x_height,
                            x_width, k_height, k_width, chans_in, chans_out, h_stride,
                            v_stride, seed, larq_clamp_min, larq_clamp_max, valid_impl);
                      }
                    }

                    free(X_ref);
                    free(Y);
                    free(Y_ref);
                    free(post_activation_multiplier_q);
                    free(post_activation_bias_q);
                    free(quantised_accu_modifier);
                    free(K);
                    free(K_ref);

                    free(post_activation_multiplier);
                    free(post_activation_bias);
                    free(chan_overlaps);
                }
              }
            }
          }
        }
      }
    }
  }
}


void impl_bconv2d_int8_pseudo_random2(
  const unsigned max_x_height, const unsigned max_x_width,  
  
  const unsigned chans_in,

  const unsigned min_chans_out,
  const unsigned max_chans_out,

  const unsigned chans_in_inc, 
  const unsigned chans_out_inc,

  void (* valid_impl)()) {

  for (unsigned x_height = 1; x_height <= max_x_height;
        ++x_height) {
    for (unsigned x_width = 1; x_width <= max_x_width;
          ++x_width) {
      unsigned k_height = x_height;
      unsigned k_width = x_width;
      unsigned y_height =  CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, 1);
      unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, 1);

        for (unsigned chans_out = min_chans_out;
              chans_out <= max_chans_out; chans_out += chans_out_inc) {

          unsigned chan_words_in = chans_in/32;

          size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out*k_height*k_width*chan_words_in);
          bnn_b32_t * K_ref = (bnn_b32_t *) malloc(K_ref_bytes);
          int32_t over_bytes = compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
          bnn_b32_t * K     = (bnn_b32_t *) malloc(K_ref_bytes + over_bytes);

          size_t X_ref_bytes = sizeof(bnn_b32_t)*(x_height*x_width*chan_words_in+X_REF_OVERREAD_WORDS);
          bnn_b32_t * X_ref =(bnn_b32_t *)malloc(X_ref_bytes);
          int16_t *post_activation_multiplier_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          int16_t *post_activation_bias_q = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          bnn_b32_t *data_scratch = (bnn_b32_t *)malloc(sizeof(bnn_b32_t)*(k_height * k_width * chan_words_in + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
          
          int16_t *quantised_accu_modifier = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          
          float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
          float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
          int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

          int8_t * Y     = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);
          int8_t * Y_ref = (int8_t *) malloc(sizeof(int8_t) * y_height * y_width * chans_out);

          assert(X_ref);
          assert(Y);
          assert(Y_ref);
          assert(post_activation_multiplier_q);
          assert(post_activation_bias_q);
          assert(quantised_accu_modifier);
          assert(K);
          assert(K_ref);

          assert(post_activation_multiplier);
          assert(post_activation_bias);
          assert(chan_overlaps);

          // printf("k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u chans_out:%u\n", 
          //    k_height, k_width, x_height, x_width, chans_in, chans_out);

          int seed = 42;

          for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
            ((int*)X_ref)[b] = pseudo_rand(&seed);

          
          for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
            ((int*)K_ref)[b] = pseudo_rand(&seed);

          unsigned receptive_volume = k_width * k_height * chans_in;
          pick_post_activation_params(post_activation_multiplier, post_activation_bias, chans_out, receptive_volume, &seed);

          for (unsigned c=0;c<1024;c++){
            int32_t larq_clamp_min = pseudo_rand(&seed) % (2*receptive_volume);
            int32_t larq_clamp_max = larq_clamp_min + pseudo_rand(&seed) % (2*receptive_volume);


            run_int8_config(
                (int8_t*)Y, (int8_t*)Y_ref, (bnn_b32_t*)X_ref,
                (bnn_b32_t*)K, (bnn_b32_t*)K_ref,
                (float*)post_activation_multiplier,
                (float*)post_activation_bias, 
                (int16_t*)post_activation_multiplier_q,
                (int16_t*)post_activation_bias_q,  
                (int16_t*)quantised_accu_modifier,
                (int*) chan_overlaps,
                x_height,
                x_width, k_height, k_width, chans_in, chans_out, 1,
                1, seed, larq_clamp_min, larq_clamp_max, valid_impl);
          }
        free(X_ref);
        free(Y);
        free(Y_ref);
        free(post_activation_multiplier_q);
        free(post_activation_bias_q);
        free(quantised_accu_modifier);
        free(K);
        free(K_ref);

        free(post_activation_multiplier);
        free(post_activation_bias);
        free(chan_overlaps);
      }
    }
  }
}

static void run_int8_sub_image(
              int8_t* Y_p, 
              const int8_t* Y_ref_p, 
              const bnn_b32_t* X_p,
              const bnn_b32_t* K_p, 

              int16_t * post_activation_multiplier_q,
              int16_t * post_activation_bias_q,

              int16_t * quantised_accu_modifier,
              int16_t low_clamp_offset,
              int16_t high_clamp_offset,
              
              const int accu_shr,
              const int16_t bias_multiplier,
              const int final_shr,

              const nn_image_params_t* x,
              const nn_image_params_t* y,
              const nn_window_params_t* k,
              unsigned y_loc_x, unsigned y_loc_y, 
              unsigned y_sub_width, unsigned y_sub_height,
              void (* valid_impl)()){

  valid_impl(Y_p, X_p,
      K_p, post_activation_multiplier_q,
      post_activation_bias_q, accu_shr, bias_multiplier, final_shr, 
      quantised_accu_modifier, low_clamp_offset, high_clamp_offset,

      x, y, k,
      y_loc_x, y_loc_y, y_sub_width, y_sub_height);

  int8_t(*Y)[y->width][y->channels] =
      (int8_t(*)[y->width][y->channels])Y_p;

  int8_t(*Y_ref)[y->width][y->channels] =
      (int8_t(*)[y->width][y->channels])Y_ref_p;

  for (unsigned h = 0; h < y->height; h++) {
    for (unsigned w = 0; w < y->width; w++) {
      if((h >= y_loc_y) && (h < (y_loc_y + y_sub_height)) && (w >= y_loc_x) && (w < (y_loc_x + y_sub_width))){
        //If the result should have been computed then check it against the reference
        for (unsigned c = 0; c < y->channels; c++) {
          TEST_ASSERT_INT8_WITHIN(1, Y_ref[h][w][c], Y[h][w][c]);
        }
      } else {
        //Otherwise check thet is hasn't been written to
        for (unsigned c = 0; c < y->channels; c++) {
          TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
        }
      }
    }
  }
}

/*
This test check for a fixed x_height, x_width, k_height and k_width a sub-region of the output
is correctly computed. It check this for MIN_CHANS_IN and MAX_CHANS_IN input channels and 
MIN_CHANS_OUT to MAX_CHANS_OUT output channels. Stride are tested, dilations are untested currently.
*/
void impl_bconv2d_int8_sub_image(
  const unsigned full_x_height, const unsigned full_x_width,  
  const unsigned full_k_height, const unsigned full_k_width,
  
  const unsigned min_chans_in, const unsigned max_chans_in,    
  const unsigned min_chans_out, const unsigned max_chans_out,  

  const unsigned chans_in_inc, const unsigned chans_out_inc,

  const unsigned min_v_stride, const unsigned max_v_stride, 
  const unsigned min_h_stride, const unsigned max_h_stride,
  void (* valid_impl)()){

  #define X_V_DILATION 1
  #define X_H_DILATION 1

  int seed = 42;

  for(unsigned chans_out = min_chans_out; chans_out <= max_chans_out; chans_out += chans_out_inc){
    for(unsigned chans_in = min_chans_in; chans_in <= max_chans_in; chans_in += chans_in_inc){

      unsigned chan_words_in = chans_in/32;

      size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out*full_k_height*full_k_width*chan_words_in);
      bnn_b32_t * K_ref = (bnn_b32_t * ) malloc(K_ref_bytes);

      size_t X_ref_bytes = sizeof(bnn_b32_t)*(full_x_height*full_x_width*chan_words_in+X_REF_OVERREAD_WORDS);
      bnn_b32_t * X_ref = (bnn_b32_t *) malloc(X_ref_bytes);

      int16_t * post_activation_multiplier_q = (int16_t *) malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
      int16_t * post_activation_bias_q = (int16_t *) malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));

      int16_t *quantised_accu_modifier = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          
      int32_t over_bytes = compute_int8_over_RW_bytes(chans_in, full_k_height, full_k_width, chans_out);

      bnn_b32_t * K = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(chans_out*full_k_height*full_k_width*chan_words_in) + over_bytes);

      float * post_activation_multiplier = (float *)malloc(sizeof(float)*chans_out);
      float * post_activation_bias = (float *)malloc(sizeof(float)*chans_out);
      int * chan_overlaps = (int *)malloc(sizeof(int)*(chans_out));

      for (unsigned h_stride = min_h_stride; h_stride < max_h_stride; h_stride++){
        for (unsigned v_stride = min_v_stride; v_stride < max_v_stride; v_stride++){
            
          nn_image_params_t x;
          x.height = full_x_height;
          x.width = full_x_width;
          x.channels = chans_in;
          nn_image_params_t y;
          y.height = CONV2D_OUTPUT_LENGTH(full_x_height, full_k_height, X_V_DILATION, v_stride);
          y.width = CONV2D_OUTPUT_LENGTH(full_x_width, full_k_width, X_H_DILATION, h_stride);
          y.channels = chans_out;
          nn_window_params_t k;
          k.shape.height = full_k_height;
          k.shape.width = full_k_width;
          k.stride.horizontal = h_stride;
          k.stride.vertical = v_stride;
          k.dilation.horizontal = X_H_DILATION;
          k.dilation.vertical = X_V_DILATION;

          int8_t * Y_ref = (int8_t *) malloc(sizeof(int8_t) * y.height * y.width * y.channels);
          int8_t * Y     = (int8_t *) malloc(sizeof(int8_t) * y.height * y.width * y.channels);    

          if(y.height == 0 || y.width == 0)
            continue;

          for(unsigned i=0;i<1<<6;i++){

            for(unsigned b=0;b<X_ref_bytes/sizeof(int);b++)
              ((int*)X_ref)[b] = pseudo_rand(&seed);
            
            for(unsigned b=0;b<K_ref_bytes/sizeof(int);b++)
              ((int*)K_ref)[b] = pseudo_rand(&seed);

            unsigned receptive_volume = k.shape.width * k.shape.height * x.channels;

            pick_post_activation_params(post_activation_multiplier, post_activation_bias, chans_out, receptive_volume, &seed);

            //Calculate the entire reference image
            larq_ref_bconv2d_int8_out(&x, &y, &k, (const int32_t*)X_ref, (const int32_t*)K_ref,
                        (int8_t*)Y_ref, post_activation_multiplier, post_activation_bias, 0, INT_MAX);

            //TODO modulate these
            int32_t larq_clamp_min = 0;     
            int32_t larq_clamp_max = receptive_volume*2;

            bnn_reorder_kernel_tensor((bnn_b32_t *)K, (const bnn_b32_t *)K_ref, k.shape.height, 
              k.shape.width, x.channels, y.channels, chan_overlaps);

            int accu_shr, final_shr;
            int16_t bias_multiplier;

            int16_t low_clamp_offset;
            int16_t high_clamp_offset;

            bnn_quantise_activation(
                post_activation_multiplier_q,
                post_activation_bias_q,

                post_activation_multiplier,
                post_activation_bias, 

                chans_out,

                larq_clamp_min, 
                larq_clamp_max,

               quantised_accu_modifier,
               &low_clamp_offset,
               &high_clamp_offset,

                &accu_shr, &bias_multiplier, &final_shr, receptive_volume, chan_overlaps
            );

            for (unsigned y_loc_x = 0; y_loc_x<y.width; ++y_loc_x){
              for (unsigned y_loc_y = 0; y_loc_y<y.height; ++y_loc_y){
                for (unsigned y_sub_width = 1; y_sub_width<y.width-y_loc_x; ++y_sub_width){
                  for (unsigned y_sub_height = 1; y_sub_height<y.height-y_loc_y; ++y_sub_height){

                      size_t addressable_Y_bytes = y.height * y.width * y.channels;
                      memset(Y, undef_sentinal, addressable_Y_bytes);

                      for (unsigned c=0;c<1024;c++){
                        int32_t larq_clamp_min = pseudo_rand(&seed) % (2*receptive_volume);
                        int32_t larq_clamp_max = larq_clamp_min + pseudo_rand(&seed) % (2*receptive_volume);

                        run_int8_sub_image(
                          (int8_t*)Y, 
                          (const int8_t*)Y_ref,
                          (const bnn_b32_t*) X_ref,
                          (const bnn_b32_t*) K, 

                          (int16_t * )post_activation_multiplier_q,
                          (int16_t *) post_activation_bias_q,
                          (int16_t *) quantised_accu_modifier,
                          low_clamp_offset,
                          high_clamp_offset,

                          (const int )accu_shr,
                          (const int16_t) bias_multiplier,
                          (const int )final_shr,

                          &x, &y, &k,
                          y_loc_x, y_loc_y, y_sub_width, y_sub_height, valid_impl);
                      }
                    }
                  }
                } 
              }
            }
            free(Y_ref);
            free(Y);
        }
      }
      free(K_ref);
      free(K);
      free(X_ref);
      free(post_activation_multiplier);
      free(post_activation_bias);
      free(chan_overlaps);
      free(post_activation_multiplier_q);
      free(post_activation_bias_q);
      free(quantised_accu_modifier);

    }
  }
}


void impl_bconv2d_int8_directed(void (*valid_impl)()) {
  const unsigned h_stride = 2, v_stride = 2;
  const unsigned k_height = 3, k_width = 3;
  const unsigned x_height = 9, x_width = 9;
  const unsigned y_height = 4, y_width = 4;
  const unsigned chans_in = 32, chans_out = 64;

  const unsigned receptive_volume = k_height * k_width * chans_in;
  const unsigned chan_words_in = chans_in / 32;

  const unsigned K_ref_size = chans_out * k_height * k_width * chan_words_in;
  const bnn_b32_t K_ref[K_ref_size] = {
      1539622939,  1523638586,  2144401211,  -394156741,  1721840443,
      125909467,   -1534772101, -1548935457, 589637337,   -1827854441,
      -1579770841, 1876088943,  -1856096107, -321758107,  1859275883,
      -770818626,  1952103723,  1826239786,  1210442292,  1210409600,
      270878344,   -1872379228, 3228352,     -2034250787, -1531620727,
      -2034675239, -554307077,  1806468858,  1403356922,  -668694790,
      1533068922,  1524942458,  1246021242,  1522585194,  1254147706,
      1852492922,  -418999809,  -191581837,  -61562591,   -1865047519,
      -1207623168, 3083905,     1012993537,  1682590860,  -3224225,
      1540546816,  533883136,   -1663677660, 180414496,   465750050,
      534563122,   683813892,   180374534,   401623530,   -1483800593,
      -448889619,  -448857969,  1199887789,  1200543917,  -1550921331,
      1384310732,  327607276,   -2088311891, 1447072214,  -2137016850,
      -2011267991, 792726552,   -18869866,   -1801473561, 543058977,
      2069909009,  -2074218,    -36679310,   -198962797,  -1609264140,
      1343628420,  -1859311743, -2127747452, -1030314199, -1835096648,
      -1768249351, -71110538,   1808980094,  -84764290,   -339684994,
      -271396354,  -3083906,    -339422257,  -1981467687, 409035105,
      -1878252656, -1876020348, -1859312748, -1874712496, 281935956,
      -255449068,  2035959894,  835340356,   2035998326,  -313126665,
      -199523606,  1618035408,  -316780417,  -451194773,  -1401330725,
      -329641910,  -467930837,  -1539316821, -1522002737, -700441459,
      1520558920,  -451013491,  -163570483,  -622727912,  -451012355,
      -164621171,  -631134776,  1177096986,  1177125690,  606700475,
      -752241990,  1395238842,  2058200890,  1932394431,  1537583098,
      1537582075,  475008000,   5711361,     741936769,   102239488,
      -2078653371, -2079717851, 531623380,   -1799173531, -2136487227,
      -1609592627, -529420127,  -634265430,  -1607216955, -524438805,
      -630857301,  -127561519,  -1868189213, -625098837,  639501517,
      1698240173,  1681856191,  2000492447,  -1540548705, 1631426439,
      -12746855,   -1221740647, -1280349296, -2821830,    2042943802,
      2030852756,  1473181994,  2144272170,  1036975401,  924758971,
      526957355,   2001484681,  -760958017,  -147998785,  -165104641,
      -1163020361, -416434241,  -1238517827, -1108275388, -1182986683,
      -2114606524, 2130710811,  -149908993,  -410076673,  134744577,
      218436497,   -8121445,    -1862357407, 4140545,     1302662657,
      801298779,   -170735254,  -1859311723, 785565809,   -187545234,
      -1859278955, 785303627,   2031532152,  -1859278915, 656290953,
      798645257,   769526829,   260742186,   1338694762,  1591385198,
      -1291544851, 428250350,   -2121361953, 1043152795,  -5900389,
      -328337957,  -1499776663, -1498726487, -1348813399, 730516076,
      -2104916409, 42631531,    -44789646,   -45035402,   -45039277,
      2030874811,  2030348338,  2072291355,  1361919635,  2133605891,
      2135693339,  -401017847,  1859812649,  1067079099,  -392649599,
      1054371113,  1071148313,  -1028807477, -1731678935, 514666394,
      -1065049497, -1466458013, -1596328661, -666525114,  -2002803613,
      -1533346261, -924475164,  -930175961,  -1265041877, 1577453005,
      1578527048,  1586389996,  -1898436336, -2032654284, -2033670092,
      -1446470601, -1316447178, -505110986,  2958976,     271394465,
      942417562,   -341922177,  -342180033,  -620053057,  -2975362,
      -2966146,    -140722690,  -787667016,  -1844632432, -16740037,
      -1878186072, -1842534480, -1559457413, -1874908543, -2136136910,
      -1509997961, -670222871,  1378828185,  1244610267,  -808797826,
      -270886530,  -270870148,  1058281365,  -48983787,   -111886043,
      319071911,   1535545399,  1468194155,  285517732,   1535410614,
      1532019002,  -1859377232, 1461490588,  1578938683,  1858767226,
      -959805316,  -673538967,  1997836598,  -183201410,  -116092813,
      406679169,   -1741871487, -1707266431, 830439301,   893271941,
      -176209279,  272514177,   320486401,   -1306551263, 349596481,
      1071278405,  -1697219132, -1341748106, 1631591492,  1684674630,
      956730470,   1766231118,  1796660298,  419925710,   1095667340,
      -1822753332, 415263744,   478424854,   -1525804033, 448801792,
      -1616565453, -1489169409, 448965696,   -1112206473, -146926593,
      750929984,   -321763260,  -120428476,  1020418049,  751978582,
      -53322649,   751983619,   768756807,   -1395458841, -1609956347,
      -2130139773, -539183685,  -1593177052, -2142763904, -1713587039,
      760925284,   744148044,   -1940206559, -712992577,  288206469,
      261527671,   1438820396,  271429252,   1338284115,  1448454617,
      271394692,   -1496008325, -2134727710, 8553446,     1667367894,
      -61587054,   -129744558,  1969488854,  -47894253,   2103706131,
      1970084759,  -1849382362, -1778076153, 1195679237,  -783960409,
      -785537403,  1363973765,  -183714641,  1965332111,  1359855277,
      1612595396,  2034909418,  957236966,   -27471384,   1591363946,
      1574888366,  -696665831,  1587063162,  1589307770,  -1069124890,
      -1605987585, -406415137,  -1072295707, -535424779,  -720954673,
      -1067888425, -1056517401, -975908105,  656033865,   1719812304,
      -457158425,  1191856221,  1086341204,  1539625686,  -161480196,
      1480608460,  1533397718,  -946276870,  -569542722,  825044612,
      -19885702,   -560192088,  288304772,   -137310121,  -694509497,
      -1320210783, 272763972,   272760836,   272662548,   -62605421,
      -198363757,  2118497687,  -291586565,  -1348821509, -409297415,
      -568979131,  -704235187,  -586270243,  -559015739,  -1699792187,
      -1733313707, -56124211,   -621957563,  -1729873211, -129990366,
      657137032,   -1011893799, 791353865,   -944809508,  -657071821,
      128850389,   -120201165,  -1135317469, -2137545879, 1478494720,
      2137328662,  -896421533,  -1998378455, 224464409,   -825183909,
      -1932263829, -844293029,  -797597151,  -2143536127, 1739200989,
      70200961,    128604569,   125835741,   129642153,   130709977,
      -1614653090, -2033704624, -2046287532, -2041026224, -238503883,
      -1861437419, -1773610921, 2034695338,  1766235275,  2076499198,
      1388349504,  1388505090,  1531112192,  1522974570,  1522903722,
      1383050160,  -762006037,  1517633401,  373238673,   2002786712,
      -1876086316, -2066304925, 1734352281,  -1876090732, -1936285085,
      2145541913,  1387070652,  -1465997781, 53254612,    -455823069,
      925523340,   -762011369,  -1395491799, 931290588,   -862607309,
      -1483715511, 1408650693,  -157834882,  -1876056187, 1183486291,
      -57484949,   -1859377535, 1863489915,  -49242774,   288106129,
      2066259067,  -1745405552, -674045541,  -2074221659, -942378535,
      -674176526,  -1747522574, -825103494,  -674176014,  -976950282,
      1875516697,  1465915800,  -796458028,  -1534149589, 1801985433,
      1533346772,  -1600865245, 801651977,   2078623132,  -251342594,
      1896137943,  1830078419,  403580041,   -1774312055, -1633800311,
      -2034198404, 786102654,   -1883076482, -1500857045, -1497711191,
      -1499775621, 50988740,    -2146859324, -2045860266, -608025898,
      1212564052,  1213096534,  2086034500,  -61731836,   -61469631,
      1342848286,  269237300,   -1828014298, -678652930,  -476933122,
      -1013796353, 608844809,   2829965,     321851284,   794430465,
      1059729304,  1005526034,  459935232,   963419683,   963493391,
      -696208945,  -2145410007, 1633753104,  1474320793,  -2070548029,
      -1397266903, -1357095911, -4173488,    -1801984017, -828719254,
      1393208958,  1326091390,  -682609666,  -750865410,  -1018283137,
      794888853,   -676514849,  -1797803145, -2110937471, 264427899,
      -52764177,   -1842534779, 1876089211,  -170754778,  304981649,
      1876056423,  -246223827,  873078345,   -2026989092, -254499981,
      1202592220,  -791305933,  -1127207389, -254435021,  -1133441495,
      730549468};

  int32_t over_bytes =
      compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
  bnn_b32_t *K =
      (bnn_b32_t *)malloc(K_ref_size * sizeof(bnn_b32_t) + over_bytes);

  const unsigned X_ref_size = x_height * x_width * chan_words_in;
  const bnn_b32_t X_ref[X_ref_size + X_REF_OVERREAD_WORDS] = {
      -1837422508, 276392448,   1516857880,
      317303634,   -1698321546, -2099171372,
      -2098484012, 181661554,   0,
      -221120386,  450455632,   1522658170,
      1541524350,  -271461518,  1868645146,
      1878250718,  148104050,   0,
      1876122748,  -822346382,  1341780858,
      2143374202,  1876023679,  2146519358,
      1801573972,  -757992590,  0,
      2143379290,  1525799294,  -355338446,
      800152376,   -271327901,  800213434,
      1395902190,  446030778,   0,
      802905656,   -271459458,  1608635262,
      -1968326,    -337537026,  -1410223114,
      -656300166,  -1966493834, 0,
      -1535386002, 2144993075,  1060969887,
      -1524031753, -1184543018, -1182986506,
      -1318228266, -512141617,  0,
      -327328569,  -1342472321, 560946902,
      825167606,   963595910,   560946886,
      -395878657,  -482615603,  0,
      1698651789,  2101305229,  1967095695,
      -784335185,  -314614099,  -448799505,
      1832804009,  -415244595,  0,
      0,           0,           0,
      0,           0,           0,
      0,           0,           0};

  const float post_activation_multiplier_original[chans_out] = {
      0.012351940385997295, 0.01575915887951851,  0.008943307213485241,
      0.010224299505352974, 0.01991458423435688,  0.008564063347876072,
      0.013291306793689728, 0.014130263589322567, 0.014040169306099415,
      0.008283263072371483, 0.012238231487572193, 0.00907240342348814,
      0.017485272139310837, 0.011195400729775429, 0.008301754482090473,
      0.01243166159838438,  0.011653591878712177, 0.013148283585906029,
      0.013521087355911732, 0.016607023775577545, 0.014706429094076157,
      0.013818399049341679, 0.010542008094489574, 0.01690087839961052,
      0.012508699670433998, 0.014033292420208454, 0.020144738256931305,
      0.016180073842406273, 0.013261908665299416, 0.017590731382369995,
      0.016404960304498672, 0.016385167837142944, 0.009051515720784664,
      0.012140316888689995, 0.017523031681776047, 0.009600243531167507,
      0.014154208824038506, 0.01888112537562847,  0.011789664626121521,
      0.0131512600928545,   0.012670175172388554, 0.01303008757531643,
      0.010212906636297703, 0.010999965481460094, 0.02007177844643593,
      0.02180858887732029,  0.013997333124279976, 0.011472060345113277,
      0.010532037355005741, 0.026670731604099274, 0.008986983448266983,
      0.016481904312968254, 0.02010033093392849,  0.021539287641644478,
      0.01189776323735714,  0.012550276704132557, 0.020099526271224022,
      0.010953588411211967, 0.018284793943166733, 0.0078839510679245,
      0.01716633513569832,  0.015060598962008953, 0.01391292829066515,
      0.017445214092731476};
  const float post_activation_bias_original[chans_out] = {
      -0.05970403924584389,  -0.03189457207918167,  -0.10995983332395554,
      -0.1819787621498108,   -0.08313516527414322,  -0.14007125794887543,
      -0.06474795192480087,  -0.05543561652302742,  -0.04212765023112297,
      -0.11703118681907654,  -0.08005634695291519,  -0.09276312589645386,
      -0.040493838489055634, -0.08497308194637299,  -0.10957623273134232,
      -0.06009465456008911,  -0.08109637349843979,  -0.06643685698509216,
      -0.14931024610996246,  -0.060489218682050705, -0.05786073952913284,
      -0.0758678987622261,   -0.08722170442342758,  -0.0533980168402195,
      -0.05867760628461838,  -0.0976688489317894,   -0.06088452786207199,
      -0.08050064742565155,  -0.047402285039424896, -0.07298173755407333,
      -0.042324043810367584, -0.0965159684419632,   -0.0762302428483963,
      -0.08213816583156586,  -0.037101179361343384, -0.08617115020751953,
      -0.06318537890911102,  -0.02164478600025177,  -0.08504659682512283,
      -0.06211598962545395,  -0.04860047996044159,  -0.05786619707942009,
      -0.0720161572098732,   -0.0612344816327095,   -0.04064817726612091,
      -0.037796907126903534, -0.059520620852708817, -0.05487140640616417,
      -0.06249295920133591,  -0.0308750718832016,   -0.06349707394838333,
      -0.06473486870527267,  -0.02874387428164482,  -0.029432030394673347,
      -0.062471356242895126, -0.07812318205833435,  -0.05101069062948227,
      -0.08313548564910889,  -0.036270491778850555, -0.08491639792919159,
      -0.03377199172973633,  -0.09227605909109116,  -0.038112759590148926,
      -0.06483688950538635};

  float output_scale = 0.0235294122248888;
  float output_zero_point = 0;
  float backtransform_add = receptive_volume;

  float post_activation_multiplier[chans_out];
  float post_activation_bias[chans_out];
  for (int j = 0; j < chans_out; j++) {
    const float post_mul = post_activation_multiplier_original[j];
    const float post_bias = post_activation_bias_original[j];
    post_activation_multiplier[j] = -1 * post_mul / output_scale;
    post_activation_bias[j] =
        (post_bias + backtransform_add * post_mul) / output_scale +
        output_zero_point;
  }

  int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));
  int16_t *post_activation_multiplier_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
  int16_t *post_activation_bias_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int16_t *quantised_accu_modifier = (int16_t *)malloc(sizeof(int16_t)*(chans_out+(16 - chans_out%16)));
          
  int8_t *Y = (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);
  int8_t *Y_ref =
      (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);

  assert(X_ref);
  assert(Y);
  assert(Y_ref);
  assert(post_activation_multiplier_q);
  assert(post_activation_bias_q);
  assert(quantised_accu_modifier);
  assert(K);
  assert(K_ref);

  assert(post_activation_multiplier);
  assert(post_activation_bias);
  assert(chan_overlaps);

  for (unsigned c = 0; c < 1 << 1; c++) {
    int seed = c;

    int32_t larq_clamp_min = 0;
    int32_t larq_clamp_max = receptive_volume;

    run_int8_config(
        (int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref, (bnn_b32_t *)K,
        (bnn_b32_t *)K_ref, 
        (float *)post_activation_multiplier,
        (float *)post_activation_bias, 
        (int16_t *)post_activation_multiplier_q,
        (int16_t *)post_activation_bias_q, 

        (int16_t *)quantised_accu_modifier,

        (int *)chan_overlaps, x_height,
        x_width, k_height, k_width, chans_in, chans_out, h_stride, v_stride,
        seed, larq_clamp_min, larq_clamp_max, valid_impl);
  }

  free(Y);
  free(Y_ref);
  free(post_activation_multiplier_q);
  free(post_activation_bias_q);
  free(quantised_accu_modifier);
  free(K);
  free(chan_overlaps);
}

static void SISO_valid(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,

      const int16_t * quantised_accu_modifier,
      const int16_t low_clamp_offset,
      const int16_t high_clamp_offset,

      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k,
      unsigned y_loc_x, unsigned y_loc_y, 
      unsigned y_sub_width, unsigned y_sub_height){

  bnn_b32_t *data_scratch = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(k->shape.height * k->shape.width * 
  x->channels/32 + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
      
  bconv2d_int8_valid(Y_p, X_p,
                      K_p, post_activation_multiplier_q,
                      post_activation_bias_q, 
                      quantised_accu_modifier, low_clamp_offset, high_clamp_offset,
                      accu_shr, bias_multiplier, final_shr, 
                      data_scratch, x, y, k,
                      y_loc_x, y_loc_y, y_sub_width, y_sub_height);
  free(data_scratch);
}

static void DI_valid(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,

      const int16_t * quantised_accu_modifier,
      const int16_t low_clamp_offset,
      const int16_t high_clamp_offset,

      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k,
      unsigned y_loc_x, unsigned y_loc_y, 
      unsigned y_sub_width, unsigned y_sub_height){

  bconv2d_int8_DIDO_valid(Y_p, (const bnn_b256_t*)X_p,
        (const bnn_b256_t*)K_p, post_activation_multiplier_q,
        post_activation_bias_q, 
        low_clamp_offset, high_clamp_offset,
        accu_shr, bias_multiplier, final_shr, 
        x, y, k,
        y_loc_x, y_loc_y, y_sub_width, y_sub_height);
}


static void SISO_full(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,

      const int16_t * quantised_accu_modifier,
      const int16_t low_clamp_offset,
      const int16_t high_clamp_offset,
      
      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k){

  bnn_b32_t *data_scratch = (bnn_b32_t *) malloc(sizeof(bnn_b32_t)*(k->shape.height * k->shape.width * 
    x->channels/32 + DATA_SCRATCH_OVERREADWRITE_WORDS)); 
      
  bconv2d_int8(Y_p, X_p,
                      K_p, post_activation_multiplier_q,
                      post_activation_bias_q, 
                      quantised_accu_modifier, low_clamp_offset, high_clamp_offset,
                      accu_shr, bias_multiplier, final_shr, 
                      data_scratch, x, y, k,
                      0, 0, y->width, y->height, 0, 0);
  free(data_scratch);
}

static void DI_full(   
      int8_t* Y_p, 
      const bnn_b32_t* X_p,
      const bnn_b32_t* K_p, 

      int16_t * post_activation_multiplier_q,
      int16_t * post_activation_bias_q,

      const int16_t * quantised_accu_modifier,
      const int16_t low_clamp_offset,
      const int16_t high_clamp_offset,

      const int accu_shr,
      const int16_t bias_multiplier,
      const int final_shr,

      const nn_image_params_t* x,
      const nn_image_params_t* y,
      const nn_window_params_t* k){

  bconv2d_int8_DIDO(Y_p, (const bnn_b256_t*)X_p,
                      (const bnn_b256_t*)K_p, post_activation_multiplier_q,
                      post_activation_bias_q, 
                      low_clamp_offset, high_clamp_offset,
                      accu_shr, bias_multiplier, final_shr, 
                      x, y, k,
                      0, 0, y->width, y->height, 0, 0);
}

void test_bconv2d_int8_sub_image(){
  impl_bconv2d_int8_sub_image(5, 5, 3, 3, 32*1, 32*9, 4*1, 4*3, 32, 4, 1, 1, 3, 3, (void*)&SISO_valid);
}

void test_bconv2d_int8_DI_sub_image(){
  impl_bconv2d_int8_sub_image(5, 5, 3, 3, 256*1, 256*2, 16*1, 16*3, 256, 32, 1, 1, 3, 3, (void*)&DI_valid);
}

void test_bconv2d_int8_pseudo_random(){
  impl_bconv2d_int8_pseudo_random(1, 5,1, 5, 32*1, 32*9, 4*1, 4*3, 32, 4, 1, 3, 1, 3, (void*)&SISO_full);
}

void test_bconv2d_int8_DI_pseudo_random(){
  impl_bconv2d_int8_pseudo_random(1, 4, 1, 4, 256*1, 256*2, 32*1, 32*3, 256, 32, 1, 3, 1, 3, (void*)&DI_full);
}

void test_bconv2d_int8_pseudo_random2(){
  impl_bconv2d_int8_pseudo_random2(1, 32, 32, 4, 4, 32, 4, (void*)&SISO_full);
}

void test_bconv2d_int8_DI_pseudo_random2(){
  impl_bconv2d_int8_pseudo_random2(1, 32, 256, 32, 32, 256, 32, (void*)&DI_full);
}

void test_bconv2d_int8_directed(){
  impl_bconv2d_int8_directed((void*)&SISO_full);
}

void test_bnn_conv2d_int8() {
  UNITY_SET_FILE();

  RUN_TEST(test_bconv2d_int8_pseudo_random);
  RUN_TEST(test_bconv2d_int8_DI_pseudo_random);

  RUN_TEST(test_bconv2d_int8_pseudo_random2);
  RUN_TEST(test_bconv2d_int8_DI_pseudo_random2);

  RUN_TEST(test_bconv2d_int8_sub_image);
  RUN_TEST(test_bconv2d_int8_DI_sub_image);

  RUN_TEST(test_bconv2d_int8_directed);
  
}
