
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_op_helper.h"
#include "nn_operator.h"
#include "vpu_sim.h"

#ifndef NN_USE_REF

extern int8_t round8(float r);

extern void vect_mat_mul_int8_asm(
  const int8_t *lhs, 
  const int8_t *rhs, 
  int8_t *vpu_buffer, 
  uint32_t channel_size,     // lhs size, rhs row size
  uint32_t rhs_col_size
);

void mat_mul_real_int8_vpu(
  nn_mat_mul_real_params_t *p,
  int8_t *vpu_buf0, int8_t *vpu_buf1, int8_t *vpu_buf2,
  int8_t *lhs, int8_t* rhs, int8_t *output) {
  for (int lhs_row = 0; lhs_row < p->lhs_row_size; ++lhs_row) {
    // TODO: optimize it with vpu
    int32_t lhs_row_sum = 0;
    for (int i = 0; i < p->channel_size; ++i) {
      lhs_row_sum += lhs[lhs_row * p->channel_size + i];
    }
    for (int rhs_col = 0; rhs_col < p->rhs_col_size; rhs_col+=16) {
      int8_t* lhs_temp = &lhs[lhs_row * p->channel_size];
      int8_t* rhs_temp = &rhs[(rhs_col+16)*p->channel_size];
      int process_col = p->rhs_col_size - rhs_col;
      process_col = process_col > 16 ? 16 : process_col;

      vect_mat_mul_int8_asm(
        lhs_temp, rhs_temp, vpu_buf0, p->channel_size, process_col);

      // TODO: optimize it with vpu
      int32_t *buff_temp = (int32_t*)vpu_buf1; // cheating here, treating vD:vR as continue space
      for (int i = 0; i < process_col; ++i) {
        buff_temp[i] = 0;
        for (int j = 0; j < p->channel_size; ++j) {
          buff_temp[i] += rhs[rhs_col*p->channel_size + i*p->channel_size + j];
        }
      }
      uint16_t *vD = (uint16_t*)(&vpu_buf0[0]);
      uint16_t *vR = (uint16_t*)(&vpu_buf0[32]);
      for (int i = 0; i < process_col; ++i) {
        uint32_t uacc = (vD[i] << 16) | vR[i];
        int32_t acc = *((int32_t*)(&uacc));
        float accf = 
          ((float)acc)
          -(p->rhs_zp * lhs_row_sum)
          -(p->lhs_zp * buff_temp[i])
          +p->in_zp_sum;
        accf *= p->scale;
        accf += p->out_zp;
        output[i] = round8(accf);
      }
      output = &output[process_col];
    }
  }
}
#endif // NN_USE_REF

void mat_mul_real_int8_ref(
    nn_mat_mul_real_params_t *p,
    int8_t *vpu_buf0, int8_t *vpu_buf1, int8_t *vpu_buf2,
    int8_t *lhs, int8_t* rhs, int8_t *output)
{
    int out_index = 0;
    for (int i = 0; i < p->lhs_row_size; ++i) {
        for (int j = 0; j < p->rhs_col_size; ++j) {
            double acc = 0.0;
            for (int k = 0; k < p->channel_size; ++k) {
                int lhs_idx = i*p->channel_size+k;
                int rhs_idx = j*p->channel_size+k;
                double x = ((double)(lhs[lhs_idx]) - p->lhs_zp);
                double y = ((double)(rhs[rhs_idx]) - p->rhs_zp);
                acc += x * y;
            }
            float quantized_value = (float)acc * p->scale + p->out_zp;
            // Clamp the quantized value to int8 range
            if (quantized_value > 127.0f)
                quantized_value = 127.0f;
            else if (quantized_value < -128.0f)
                quantized_value = -128.0f;
            output[out_index++] = (int8_t)(roundf(quantized_value));
        }
    }
}

// A real mat mul here
void mat_mul_real_int8(
  nn_mat_mul_real_params_t *p,
  int8_t *vpu_buf0, int8_t *vpu_buf1, int8_t *vpu_buf2,
  int8_t *lhs, int8_t* rhs, int8_t *output) {
  
#ifdef NN_USE_REF
  mat_mul_real_int8_ref(p, vpu_buf0, vpu_buf1, vpu_buf2, lhs, rhs, output);
#else
  mat_mul_real_int8_vpu(p, vpu_buf0, vpu_buf1, vpu_buf2, lhs, rhs, output);
#endif
}