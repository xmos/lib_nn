#include <list>
#include <tuple>

#include "AggregateFn.hpp"
#include "OutputTransformFn.hpp"

extern "C" {
#include "tst_common.h"
#ifdef LOCAL_MAIN
#undef UNITY_SET_FILE
#define UNITY_SET_FILE()
#define RUN_TEST(x) x()
#define TEST_ASSERT_EQUAL(a, b)   if ((a) != (b)) {printf("Expected %08x saw %08x\n", (int) a, (int) b); errors++;}
#else
#include "unity.h"
#endif
}

using namespace nn;

/*
  Simple test to verify maxpool
*/
int Test_Max_Pool_aggr() {
  const int vpu_ring_buffer_length = 16;
  int errors = 0;

  for (int x_height = 1; x_height <= 4; ++x_height) {
    for (int x_width = 1; x_width <= 4; ++x_width) {
      for (int x_channels = 4; x_channels <= 32 * 3; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            ImageGeometry X_params(x_height, x_width, x_channels);
            WindowGeometry K_params(k_height, k_width, 1, 1, 1, 1);

            int input_tensor_overread = 32;

            alignas(4) int8_t
                T[x_height * x_width * x_channels + input_tensor_overread];

            for(int i = 0; i < x_height * x_width * x_channels; i++) {
                T[i] = (i+13)*12345;
            }

            MatMulDirectFn_DW mmd(
                X_params, K_params
            );
            mat_mul_dw_direct_params_t p = mmd.getParams();

            int ocg_count = (x_channels + vpu_ring_buffer_length - 1) /
                            vpu_ring_buffer_length;

            for (int x = 0; x < x_height - k_height + 1; ++x) {
              for (int y = 0; y < x_width - k_width + 1; ++y) {
                for (int ocg = 0; ocg < ocg_count; ++ocg) {
                  alignas(4) VPURingBuffer A;
                  memset(&A, 0xFF, sizeof(A));
                  int8_t *X_mem_ch_grp = T + ocg * 16;
                  maxpool_direct(&p, &A, X_mem_ch_grp);
                  for (int output_chan = 0;
                       output_chan < vpu_ring_buffer_length; ++output_chan) {
                    int actual_ch = output_chan + ocg * 16;

                    if (actual_ch >= x_channels) continue;

                    int32_t v = ((int8_t *)&A.vR)[output_chan];
                    int mm = -128;
                    for(int xx = 0; xx < k_height; xx++) {
                      for(int yy = 0; yy < k_width; yy++) {
                        int d = T[(xx * x_width + yy) * x_channels + output_chan + ocg*16];
                        if (d > mm) {
                          mm = d;
                        }
                      }
                    }
                    TEST_ASSERT_EQUAL(mm, v);
                  }
                  for(int i = 16 ; i < sizeof(A); i++) {
                      TEST_ASSERT_EQUAL(-1, ((int8_t *)&A.vR)[i]);
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return errors;
}

int Test_Max_Pool_ot() {
    int8_t outp[24];
    alignas(4) VPURingBuffer A;
    for(int i = 0; i <= 16; i++) {
        ((int8_t *)&A)[i] = i+8;
    }
    int errors = 0;

    for(int len_count = 1; len_count < 16; len_count++) {
        struct otfn_int8_channelwise_params_t st = {len_count, 0};
        memset(outp, 0xff, sizeof(outp));
        int8_t *f = otfn_int8_maxpool(&st, outp+4, &A, 0, NULL);
        TEST_ASSERT_EQUAL(outp + 4 + len_count, f);
        for(int k = 0; k < len_count; k++) {
            TEST_ASSERT_EQUAL(k+8, outp[4+k]);
        }
        for(int k = 0; k < sizeof(outp); k++) {
            if (k < 4 || k >= 4 + len_count) {
                TEST_ASSERT_EQUAL(-1, outp[k]);
            }
        }
    }
    return errors;
}

extern "C" void test_maxpool();
void test_maxpool() {
  UNITY_SET_FILE();
  RUN_TEST(Test_Max_Pool_aggr);
  RUN_TEST(Test_Max_Pool_ot);
}

#ifdef LOCAL_MAIN

int main(void) {
    int errors = 0;
    errors += Test_Max_Pool_aggr();
    errors += Test_Max_Pool_ot();
    if (errors != 0) printf("FAIL\n"); else printf("PASS\n");
    return errors;
}

#endif
