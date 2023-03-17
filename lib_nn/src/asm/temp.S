
// Copyright 2023 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#if defined(__VX4A__)
.globl add_elementwise_asm
add_elementwise_asm:

.globl bsign_8
bsign_8:
.globl im_to_col_valid_impl_asm
im_to_col_valid_impl_asm:
.globl mat_mul_binary_direct_impl_asm
mat_mul_binary_direct_impl_asm:
.globl mat_mul_binary_generic_impl_asm
mat_mul_binary_generic_impl_asm:
.globl mat_mul_direct_dw_impl_asm
mat_mul_direct_dw_impl_asm:
.globl mat_mul_int8_direct_impl_asm
mat_mul_int8_direct_impl_asm:
.globl mat_mul_int8_generic_impl_asm
mat_mul_int8_generic_impl_asm:
.globl output_transform_fn_binary_impl_asm
output_transform_fn_binary_impl_asm:
.globl output_transform_fn_impl_asm
output_transform_fn_impl_asm:
.globl output_transform_fn_int_channelwise_impl_asm
output_transform_fn_int_channelwise_impl_asm:
.globl output_transform_fn_int_clamped_impl_asm
output_transform_fn_int_clamped_impl_asm:
.globl pad_3_to_4_asm
pad_3_to_4_asm:
.globl vpu_memcpy_vector_ext_asm
vpu_memcpy_vector_ext_asm:
.globl vpu_memcpy_vector_int_asm
vpu_memcpy_vector_int_asm:
.globl vpu_memset32_asm
vpu_memset32_asm:
x.retsp 0
#endif

