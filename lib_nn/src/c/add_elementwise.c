// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_op_helper.h"
#include "nn_operator.h"
#include "xs3_vpu.h"

#include "vpu_sim.h"

#ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
#define CONFIG_SYMMETRIC_SATURATION_add_elementwise                            \
  CONFIG_SYMMETRIC_SATURATION_GLOBAL
#else
#ifndef CONFIG_SYMMETRIC_SATURATION_add_elementwise
#define CONFIG_SYMMETRIC_SATURATION_add_elementwise (0)
#endif
#endif

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

#define ASHR16(A, A_SHR) (((A_SHR) >= 0) ? ((A) >> (A_SHR)) : ((A) << -(A_SHR)))
#define ROUND_SHR(A, A_SHR) (((A) + (1 << ((A_SHR)-1))) >> (A_SHR))

#define MAX(A, B) (((A) >= (B)) ? (A) : (B))
#define MIN(A, B) (((A) <= (B)) ? (A) : (B))

extern const int8_t vpu_vect_0x01[32];

unsigned mkmsk(int num){
  assert((num>=0 && num <=32) && "Number to be made into a mask has to be in this range!");
  unsigned mask = 0;
  for (int i=0; i<num; i++)
  {
    mask = (mask << 1) | 1;
  }
  return mask;
}

void add_elementwise_ref (int8_t y[], const int8_t x1[], const int8_t x2[],
                         nn_add_params_t *params, const int output_start,
                         const int output_count) {
    xs3_vpu vpu_mem;
    xs3_vpu *vpu = &vpu_mem;

    // Scratch for temp vectors
    int16_t vec_tmp1[16];
    int16_t vec_tmp2[16];

    // The number of elements VLMACC can process in int8 mode
    const int32_t vpu_epv = VPU_INT8_ACC_PERIOD;
    int multiple_of_16_count = output_count >> 4;
    int remaining_elements = output_count & 0xF;

    int index = output_start;

    int i = multiple_of_16_count;
    while(i > 0){
      unsigned mask = mkmsk(16);

      VCLRDR(vpu);
      VSETC(vpu, MODE_S8);
      VLDC(vpu, vpu_vect_0x01);
      VLMACC(vpu, &x1[index]);
      VSTR(vpu, vec_tmp1);
      VCLRDR(vpu);
      VLMACC(vpu, &x2[index]);
      VSTR(vpu, vec_tmp2);
      
      VLDR(vpu, params->bias_lo);
      VLDD(vpu, params->bias_hi);

      VSETC(vpu, MODE_S16);
      VLDC(vpu, vec_tmp1);
      VLMACC(vpu, params->m1);

      VLDC(vpu, vec_tmp2);
      VLMACC(vpu, params->m2);

      VSETC(vpu, MODE_S8);
      VLSAT(vpu, params->shift);
      VSTRPV(vpu, &y[index], mask);

      index += vpu_epv;
      i = i - 1;
    }

    // Process remaining elements
    union{
      int16_t s16[2];
      int32_t i32;
    } n;
    n.s16[0] = params->bias_lo[0];
    n.s16[1] = params->bias_hi[0];

    for (int i = 0; i < remaining_elements; i++) {
      int32_t acc = n.i32;

      acc += x1[index] * params->m1[0];
      acc += x2[index] * params->m2[0];

      acc = ROUND_SHR(acc, params->shift[0]);

      acc = MIN(acc, VPU_INT8_MAX);
      acc = MAX(acc, NEG_SAT_VAL);

      y[index] = (int8_t)acc;
      index++;
    }
}

void add_elementwise(int8_t Y[], const int8_t X0[], const int8_t X1[],
                     nn_add_params_t *p, const int output_start,
                     const int output_count) {
#ifdef NN_USE_REF
  add_elementwise_ref(Y, X0, X1, p, output_start, output_count);
#else
  add_elementwise_asm(Y, X0, X1, p, output_start, output_count);
#endif // NN_USE_REF
}