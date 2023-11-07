
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_op_helper.h"
#include "nn_operator.h"
#include "vpu_sim.h"

static int32_t clamp(int32_t v, int32_t lo, int32_t hi){
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void mul_elementwise_asm(int8_t* in1_data, int8_t* in2_data, int element_count, nn_mul_params_t * params, int8_t * out_data);

static const unsigned vlsat_shr = 1;
static const unsigned vlmul_shr = 14;
static const unsigned vdepth8_shr = 8;

void mul_boggle(nn_mul_params_t * params, 
    double in1Scale, 
    double in2Scale, 
    double outputScale,
    int8_t in1ZeroPoint,
    int8_t in2ZeroPoint, 
    int8_t outputZeroPoint){

    params->in1_zero_point = -(int32_t)in1ZeroPoint - 1;
    params->in2_zero_point = -(int32_t)in2ZeroPoint - 1;
    double scaleRatio = in1Scale * in2Scale / outputScale;
    double biasTerm = outputZeroPoint;
    
    int scalar_works = 0;
    int bias_works = 0;
    int vlashr_shr = -1;

    int32_t acc1 = (INT8_MAX - params->in1_zero_point) * (INT8_MAX - params->in2_zero_point);
    int32_t acc2 = (INT8_MIN - params->in1_zero_point) * (INT8_MIN - params->in2_zero_point);
    int32_t max_accu = (acc1 > acc2) ? acc1 : acc2;
    int32_t min_accu = (acc1 > acc2) ? acc2 : acc1;
    max_accu = (max_accu + 1) >> 1;
    min_accu = (min_accu + 1) >> 1;

    while((scalar_works == 0) || (bias_works == 0)){

        int32_t s = round(scaleRatio * pow(2, vlsat_shr + vlmul_shr + vdepth8_shr + vlashr_shr)); 
        int32_t max_product = (max_accu * s + (1<<(vlmul_shr-1))) >> vlmul_shr;
        int32_t min_product = (min_accu * s + (1<<(vlmul_shr-1))) >> vlmul_shr;

        params->scalar = s;
        if((s != params->scalar) || ((int16_t)max_product != max_product) || ((int16_t)min_product != min_product)) {
            vlashr_shr--;
            continue;
        } else {
            scalar_works = 1;
        }

        int32_t b = round(biasTerm * pow(2, vdepth8_shr + vlashr_shr));
        params->bias = b;

        int32_t max_sum = (max_product + b);
        int32_t min_sum = (min_product + b);

        if((b != params->bias) || ((int16_t)max_sum != max_sum) || ((int16_t)min_sum != min_sum)) {
            vlashr_shr--;
        } else {
            bias_works = 1;
        }
        params->vlashr_shr = vlashr_shr;
    }
}

void mul_elementwise_ref(int8_t* in1_data, int8_t* in2_data, int element_count, nn_mul_params_t * params, int8_t * out_data)
{
  for (int i = 0; i < element_count; i++) {

    int32_t accu = (int32_t)params->in2_zero_point * (int32_t)in1_data[i] +
               (int32_t)params->in1_zero_point * (int32_t)in2_data[i] +
               (int32_t)in1_data[i] * (int32_t)in2_data[i] +
               (int32_t)params->in1_zero_point * (int32_t)params->in2_zero_point +
               (int32_t)1 * (int32_t)in1_data[i] +
               (int32_t)1 * (int32_t)in2_data[i] + 
               (int32_t)1 * (int32_t)params->in1_zero_point +
               (int32_t)1 * (int32_t)params->in2_zero_point ;//+ (int32_t)1 * (int32_t)-1;

    int32_t vlsat_output = (accu + (1 << (vlsat_shr - 1))) >> vlsat_shr;  //no saturation could have happened

    if ((int16_t)vlsat_output != vlsat_output) printf("Error vlsat_output %d %d\n", (int16_t)vlsat_output, vlsat_output);
    int32_t vlmul_output = ((int32_t)params->scalar * vlsat_output + (1 << (vlmul_shr - 1))) >> vlmul_shr; //no saturation could have happened

    if ((int16_t)vlmul_output != vlmul_output) printf("Error vlmul_output %d %d -> %d\n", (int16_t)vlmul_output, vlmul_output, i);
    int32_t vladd_output =  clamp(vlmul_output + (int32_t)params->bias, INT16_MIN+1, INT16_MAX);
    int32_t vlashr_output;

    if(params->vlashr_shr > 0){
        vlashr_output = clamp((vladd_output) >> params->vlashr_shr, INT16_MIN+1, INT16_MAX);
    } else {
        vlashr_output = clamp(vladd_output << (-params->vlashr_shr), INT16_MIN+1, INT16_MAX);
    }
    
    int32_t vdepth8_output = clamp((vlashr_output +  (1 << (vdepth8_shr-1))) >> vdepth8_shr, INT8_MIN, INT8_MAX);
    out_data[i] = (int8_t)vdepth8_output;
  }
}

void mul_elementwise(int8_t* in1_data, int8_t* in2_data, int element_count, nn_mul_params_t * params, int8_t * out_data){
#ifdef NN_USE_REF
   mul_elementwise_ref(in1_data, in2_data, element_count, params, out_data);
#else
   mul_elementwise_asm(in1_data, in2_data, element_count, params, out_data);
#endif // NN_USE_REF
}
