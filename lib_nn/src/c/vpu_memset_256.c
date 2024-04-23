#include <stdint.h>
#include <string.h>
#include "vpu_memset_256.h"

#ifdef NN_USE_REF
void vpu_memset_256(void * dst, const void * src, unsigned byte_count) {
    int s = ((int) dst) & 3;
    for(int i = 0; i < byte_count; i++) {
        ((uint8_t *)dst)[i] = ((uint8_t *)src)[s];
        s = (s + 1) & 31;
    }
}
#endif

void broadcast_32_to_256(void *dst, uint32_t from) {
#ifdef NN_USE_REF
    for(int i = 0; i < 8; i++) {
        ((uint32_t *)dst)[i] = from;
    }
#else
    asm("std %0, %1, %2[0]" :: "r" (from), "r" (from), "r" (dst));
    asm("std %0, %1, %2[1]" :: "r" (from), "r" (from), "r" (dst));
    asm("std %0, %1, %2[2]" :: "r" (from), "r" (from), "r" (dst));
    asm("std %0, %1, %2[3]" :: "r" (from), "r" (from), "r" (dst));
#endif
}
