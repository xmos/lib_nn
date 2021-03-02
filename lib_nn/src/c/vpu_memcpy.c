#include <string.h>

#ifdef NN_USE_REF
void vpu_memcpy(void * dst, void * src, size_t byte_count){
  memcpy(dst, src, byte_count);
}

#else

void vpu_memcpy(void * dst, void * src, size_t byte_count){
  



}
#endif // NN_USE_REF