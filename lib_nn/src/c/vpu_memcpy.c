#include <string.h>

#define VPU_MEMCPU_ASM_BYTES_PER_ITT (128)

#ifdef NN_USE_REF
void vpu_memcpy(void * dst, const void * src, size_t byte_count){
  memcpy(dst, src, byte_count);
}

#else

void vpu_memcpy_asm(void * dst, const void * src, size_t byte_count);

void vpu_memcpy(void * dst, const void * src, size_t byte_count){
  
  //The head is from the src address to the first word aligned address
  int alignment = (int)src &0x3;
  if(alignment){
    size_t head_bytes = 3 - alignment;
    byte_count -= head_bytes;
    memcpy(dst, src, head_bytes);
    dst += head_bytes;
    src += head_bytes;
  }

  //body
  int vpu_memcpy_itts = byte_count/VPU_MEMCPU_ASM_BYTES_PER_ITT;
  vpu_memcpy_asm(dst, src, vpu_memcpy_itts);
  size_t vpy_memcpy_bytes = VPU_MEMCPU_ASM_BYTES_PER_ITT*vpu_memcpy_itts;
  dst += vpy_memcpy_bytes;
  src += vpy_memcpy_bytes;
  byte_count -= vpy_memcpy_bytes;

  //tail
  size_t tail_bytes = byte_count;
  memcpy(dst, src, tail_bytes);
}
#endif // NN_USE_REF