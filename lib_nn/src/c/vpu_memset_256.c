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
