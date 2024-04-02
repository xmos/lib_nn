#include <string.h>
#include "vpu_memmove_word_aligned.h"

#ifdef NN_USE_REF
void vpu_memmove_word_aligned(void * dst, const void * src, unsigned byte_count) {
    memmove(dst, src, byte_count);
}
#endif
