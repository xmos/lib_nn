#include <xs1.h>

#if defined(__VX4A__)
#include "nn_operator.h"
#include "../src/asm/asm_constants.h"
#include "vpu_sim.h"
void bsign_8_ref(bnn_b32_t* y, const int8_t* x, const int8_t* zero_point_vect,
    const nn_bsign_8_job_t* job) ;
void bsign_8(
    bnn_b32_t *Y,
    const int8_t *X,
    const int8_t *zero_point_vect,
    const nn_bsign_8_job_t *job) {
        bsign_8_ref(Y, X, zero_point_vect, job);
    }

void expand_8_to_16(int16_t *out, int8_t *in, int N) {
    for(int i = 0; i < N; i++) {
        out[i] = in[i];
    }
}

void dequantize_int16_tensor_ref(float *output, int16_t *input, int tensor_length, void *blob);
void dequantize_int16_tensor_asm(float *output, int16_t *input, int tensor_length, void *blob) {
    dequantize_int16_tensor_ref(output, input, tensor_length, blob);
}
void multiply_int16_tensor_ref(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) ;
void multiply_int16_tensor_asm(int16_t *output, int16_t *input1, int16_t *input2, int tensor_length, void *blob) {
    multiply_int16_tensor_ref(output, input1, input2, tensor_length, blob);
}

void requantize_int16_tensor_ref(int16_t *output, int16_t *input1, int tensor_length, void *blob) ;
void requantize_int16_tensor_asm(int16_t *output, int16_t *input1, int tensor_length, void *blob) {
    requantize_int16_tensor_ref(output, input1, tensor_length, blob);
}

void output_transform_fn_int16_impl_asm(int16_t *vDvR,
                                        int32_t *mul_add,
                                        int16_t *output,
                                        uint32_t N);

void pad_3_to_4_asm(int32_t outputs[], int64_t inputs[], uint32_t N_24, uint32_t pad_val) {
    int8_t * outputs_p = (int8_t *)outputs;
    int8_t * inputs_p = inputs;
    for(uint32_t l=0;l<N_24;l++){
        for (unsigned i=0;i<8;i++){
            memcpy(outputs_p, inputs_p, 3);
            inputs_p += 3;
            outputs_p += 3;
            memcpy(outputs_p, &pad_val, 1);
            outputs_p += 1;
        }
    }
};

int8_t *output_transform_maxpool_impl_asm(
    const void *params, int8_t *Y, void *A,
    int16_t *multipliers_and_biases, int output_count) {}
int8_t *output_transform_fn_int_clamped_impl_asm(
    const void *params, int8_t *Y, void *A,
    int32_t output_channel_group, int16_t *offsets_multipliers_and_biases) {}
int8_t *output_transform_fn_binary_impl_asm(
    int8_t *Y, void *A, int32_t output_channel_group,
    int16_t *thresholds) {}
void output_transform_fn_int16_impl_asm(int16_t *vDvR,
                                        int32_t *mul_add,
                                        int16_t *output,
                                        uint32_t N) {}
void quantize_int16_tensor_asm(int16_t *output,
                               float *input, int tensor_length, void *blob) {}
void mat_mul_direct_binary_impl_asm(const void *params,
                                    void *A, int8_t *X,
                                    int32_t output_channel_group,
                                    int8_t *weights) {}
void mat_mul_generic_binary_impl_asm(const void *params,
                                     void *A, int8_t *X,
                                     int32_t output_channel_group,
                                     int8_t *weights) {}
void maxpool_direct_impl_asm(const void *params, void *A, int8_t *X) {}
void mat_mul_dw_direct_int16_impl_asm(const void *params,
                                      void *A, int16_t *X,
                                      int32_t output_channel_group,
                                      int16_t *weights) {}
int8_t *output_transform_fn_int_channelwise_impl_asm(
    const void *params, int8_t *Y, void *A,
    int16_t *multipliers_and_biases, int output_count) {}

#endif