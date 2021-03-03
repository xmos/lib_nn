#include <stdint.h>
#include "xs3_vpu.h"

#define VPU_MEMSET_ASM_WORDS_PER_ITT XS3_VPU_VREG_WIDTH_WORDS

#ifdef NN_USE_REF
void vpu_memset(void * dst, const int32_t value, const unsigned word_count){
  int32_t * dst32 = (int32_t *)dst;
  for (int i=0; i < word_count; i++)
    dst32[i] = value;
}

#else

void vpu_memset32_asm(void * dst, const int32_t value, const unsigned byte_count);

void vpu_memset(void * dst, const int32_t value, const unsigned word_count){
  
  int32_t * dst32 = (int32_t *)dst;
  
  //do the leading words
  unsigned leading_words = word_count%VPU_MEMSET_ASM_WORDS_PER_ITT;
  for (int i=0; i < word_count; i++)
    dst32[i] = value;

  dst32 += leading_words;
  
  //do the remaining multiple of VPU_MEMSET_ASM_WORDS_PER_ITT words
  vpu_memset32_asm(dst32, value, word_count / VPU_MEMSET_ASM_WORDS_PER_ITT);

}
#endif // NN_USE_REF