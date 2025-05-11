#include "nn_arch.h"

nn_target_arch_t NN_ARCH = TARGET_ARCH_XS3A;

void SetNNTargetArch(nn_target_arch_t arch) {
    NN_ARCH = arch;
}