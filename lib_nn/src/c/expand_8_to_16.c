#include "expand_8_to_16.h"

#ifdef NN_USE_REF
void expand_8_to_16(int16_t *out, int8_t *in, int N) {
    for(int i = 0; i < N; i++) {
        out[i] = in[i];
    }
}
#endif
