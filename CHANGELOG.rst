lib_nn change log
=================

0.7.0
-----

* ADDED: Quantized int8 matrix multiplication support for XS3 and VX4.
* ADDED: VPU-optimized int8 mean reduction for supported last-axis operations.
* ADDED: VX4 implementations for quantization, dequantization, requantization, int16 multiplication and additional matrix multiplication and output-transform kernels.
* ADDED: Support for the new VX4 Clang toolchain.
* ADDED: Expanded coverage for matrix multiplication, output transforms, pooling, memory helpers and VPU instructions.
* ADDED: Expanded documentation covering build instructions, kernel usage, architecture-specific behavior and API references.
* CHANGED: Consolidated layer declarations into ``nn_layers.h`` and VPU memory helpers into ``vpu_mem.h``.
* CHANGED: Improved custom CMake builds and native GCC, Clang and MSVC support.
* CHANGED: Optimized VX4 elementwise addition and multiplication by removing redundant saturation workarounds.
* FIXED: Incorrect VX4 vector-mode selection for int16-by-int8 matrix multiplication.
* FIXED: Architecture-specific rounding, arithmetic shifts and saturation in VPU simulation and output-transform calculations.
* REMOVED: Legacy pooling headers and configurable symmetric-saturation options.

0.6.0
-----

  * Initial relase

