#include <string.h>
#include <assert.h>

#include "nn_op_utils.h"

void vpu_memcpy(void * dst, const void * src, size_t byte_count){
  
  //The code below doesnt support such small copies
  if (byte_count < 4){
    memcpy(dst, src, byte_count);
    return;
  }

  //src and dst alignment must be the same
  assert(((int)dst &0x3) == ((int)src &0x3));

  //The head is from the src address to the first word aligned address
  int alignment = (int)src&0x3;
  if(alignment){
    size_t head_bytes = 4 - alignment;
    byte_count -= head_bytes;
    memcpy(dst, src, head_bytes);
    dst += head_bytes;
    src += head_bytes;
  }

  //body
  int vector_count = byte_count/VPU_MEMCPU_VECTOR_BYTES;
  vpu_memcpy_vector(dst, src, vector_count);
  size_t vpy_memcpy_bytes = VPU_MEMCPU_VECTOR_BYTES*vector_count;
  dst += vpy_memcpy_bytes;
  src += vpy_memcpy_bytes;
  byte_count -= vpy_memcpy_bytes;

  //tail
  size_t tail_bytes = byte_count;
  memcpy(dst, src, tail_bytes);
}

#ifdef NN_USE_REF

void vpu_memcpy_vector(void * dst, const void * src, int vector_count){
  memcpy(dst, src, vector_count*VPU_MEMCPU_VECTOR_BYTES);
}

#else

void vpu_memcpy_asm(void * dst, const void * src, size_t byte_count);

void vpu_memcpy_vector(void * dst, const void * src, int vector_count){

  assert(((int)dst&0x3) == 0);

  vpu_memcpy_asm(dst, src, vector_count);
}


#endif // NN_USE_REF