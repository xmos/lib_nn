// Copyright 2020 XMOS LIMITED. This Software is subject to the terms of the 
// XMOS Public License: Version 1

#include "asm_constants.h"

#include "nn_operator.h"

// This structure is deprecated. Only keeping it around because some of the assembly routines need it.
const vpu_constants_t vpu_vects = {

{   0x007F, 0x007F, 0x007F, 0x007F, 0x007F, 0x007F, 0x007F, 0x007F,
    0x007F, 0x007F, 0x007F, 0x007F, 0x007F, 0x007F, 0x007F, 0x007F },

{   0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01,
    0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01  },

{   0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02,
    0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02, 0x02  },

{   -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
    -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
    -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80,
    -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80, -0x80  },

};

const uint32_t vpu_vect_zero[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };

#if __xcore__


asm(".set vpu_vects_vec_0x007F, (vpu_vects + 0x00); .global vpu_vects_vec_0x007F");
asm(".set vpu_vects_vec_0x01,   (vpu_vects + 0x20); .global vpu_vects_vec_0x01"  );
asm(".set vpu_vects_vec_0x0002, (vpu_vects + 0x30); .global vpu_vects_vec_0x0002");
asm(".set vpu_vects_vec_0x80,   (vpu_vects + 0x50); .global vpu_vects_vec_0x80"  );


#endif
