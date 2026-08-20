##############################
lib_nn: Neural network library
##############################

************
Introduction
************

``lib_nn`` is a library of optimised kernels for the neural network operators commonly used in 8-bit quantised inference, such as convolution, pooling, fully-connected layers and elementwise operations. Each kernel is written to maximise performance and minimise memory footprint on XMOS devices.

This library targets the xs3 and vx4 architectures. These architectures have a vector unit with 256-bit wide registers that can operate in 8-bit, 16-bit or 32-bit integer mode; ``lib_nn`` kernels are hand-written to make direct use of this vector unit, alongside portable C reference implementations of the same operators.

This document assumes familiarity with the XMOS xCORE architecture, the XMOS tool chain, the C programming language, and neural network concepts.

*****
Usage
*****

``lib_nn`` is intended to be used with the `XCommon CMake <https://www.xmos.com/file/xcommon-cmake-documentation/?version=latest>`_
, the `XMOS` application build and dependency management system.

To use this library in an application include ``lib_nn`` in the application's ``APP_DEPENDENT_MODULES`` list in
`CMakeLists.txt`, for example:

.. code-block:: cmake

    set(APP_DEPENDENT_MODULES "lib_nn")

.. note:: Dependent modules should be pinned to release versions where possible, otherwise the
   latest commit on the `develop` branch will be used.  For further details on managing modules,
   pinning to a release version and other options, please see the page `xcommon-cmake Dependency Management <https://www.xmos.com/documentation/XM-015090-PC/html/doc/dependency_management.html>`_.

``lib_nn`` functions are accessed via their respective header files, for example:

.. code-block:: C

    #include "nn_pooling.h"
    #include "nn_layers.h"

*********************
Example application
*********************

The ``examples/add_tensor`` directory contains a minimal application that demonstrates how to use ``lib_nn``.

The program prints the two input tensors and the result of adding them together element-wise.

This example is built for ``XK-EVK-XU316`` target and runs it in the xcore simulator.

First, make sure the XMOS XTC tools are installed and activated.
Then, from the top level of the repository, run the following commands::

    # go to the example directory
    cd examples/add_tensor
    # build
    cmake -G "Unix Makefiles" -B build && cmake --build build
    # run (simulation)
    xsim bin/add_tensor.xe
    # >> output
    Input1: -100,   200,    300,    400,    -500,   800,    100,    -50,    -25,    1000,   1100,   1200,
    Input2: 100,    200,    300,    400,    500,    600,    700,    800,    900,    1000,   1100,   1200,
    Output: 0,      400,    600,    800,    0,      1400,   800,    750,    875,    2000,   2200,   2400,

See ``examples/add_tensor/README.rst`` for the full walkthrough.

********
Concepts
********

Networks, Operators, Instances and Jobs
========================================

The design of ``lib_nn`` centres around a concept hierarchy that breaks down as follows.

Networks
--------

At the top level of the hierarchy is the concept of a *network*. A network is a sequence of operations and the data that joins them that accomplishes some computational task, such as performing inference using a convolutional neural network. ``lib_nn`` in its raw form does not have any explicit semantic representation of a network; a network is instead created by the sequence of invocations performed by a user of ``lib_nn``.

Operators
---------

Below the network is an *operator*. An operator is an abstraction representing a certain class of operations. For example, ``avgpool2d`` is an operator that performs 2D average pooling on images by sliding a pooling window of arbitrary size in two dimensions around an input image to produce an output image. An operator is represented semantically in the API by a set of struct definitions and functions capable of performing the necessary arithmetic.

Operator Instances
-------------------

It will often be the case that a network makes use of the same operator multiple times, for example, by having alternating layers of convolutions and pooling. Each occurrence of an operator within the network has a set of hyperparameters which describe the structure of the work to be performed, such as the size of a convolution window, or the number of channels processed by a pooling operation. An operator together with its hyperparameters constitutes a concrete instance of that operator.

Jobs
----

It is often beneficial to split the actual execution of the work for an operator instance into multiple parts. This may be done, for example, to reduce latency by dividing the work among multiple cores that can run in parallel, or to reduce the memory overhead by only keeping part of the parameters or data in SRAM at a time. Each block of work to be performed is referred to as a *job*. In ``lib_nn``, each job corresponds to a subset of the data to be output by an operator instance. In some operators a job will compute a rectangular subset of an output image, while in others a job will compute a contiguous block of the output's memory.

Logical vs API Entities
------------------------

The API distinguishes between a logical tensor and its representation in
memory. A logical tensor is the mathematical object operated on; its API
representation is the pointer, shape information, and memory layout supplied
to a kernel. 

The representation is sometimes the standard tensor layout, and
sometimes an optimised layout required by the VPU.

For example, ``maxpool2d()`` uses a simple representation: ``X`` and ``Y``
are pointers to images, while ``x_params`` and ``y_params`` supply their
shapes. The header specifies row-major memory with channels innermost, so the
logical element ``X[r,c,p]`` is stored at ``(r * width + c) * channels + p``.
``maxpool2d_ext()`` uses the same image representation, with ``job_params``
selecting the output rows, columns, and channels computed by that call.

************************
Implementation Structure
************************

The following groups cover the main functional areas of the library, each
mapped to the operators and source files that implement them.

- **Pooling and image operators**: reduce an input image to an output image by sliding a window and computing a per-channel aggregate. e.g. ``maxpool2d()``, ``avgpool2d_global()``, ``argmax_16()``.
- **Convolution**: transform an input image and kernel into an output image through weight reordering, multiply-accumulate, and per-channel output scaling. e.g. ``reorder_kernel_weights()``, ``mat_mul_direct_int8()``, ``execute()``. Depthwise and transpose variants included.
- **Elementwise operators**: apply arithmetic operations element-by-element across two tensors of the same shape. e.g. ``add_elementwise()``, ``mul_elementwise()``, ``add_int16_tensor()``.
- **Quantisation / dequantisation**: convert tensors between floating-point and fixed-point representations, with a compile-time ``*_blob()`` call to pre-compute runtime parameters. e.g. ``quantize_int16_tensor()``, ``dequantize_int16_tensor_blob()``.
- **Activation and reduction**: apply non-linear functions or reduce a tensor along a dimension to a scalar output. e.g. ``softmax_generate_exp_lut()``, ``quadratic_interpolation_128()``, ``mean_int8()``.
- **Data utilities**: repack or reformat tensor data into layouts required by the VPU. e.g. ``bsign_8()``, ``expand_8_to_16()``, ``pad_3_to_4_run()``.
- **VPU utilities**: copy, move and set memory at word and vector alignment; simulate VPU instructions for C reference implementations. e.g. ``vpu_memcpy()``, ``VLMACCR()``, ``VLSAT()``.

**********************
Implementation Details
**********************

The following notes describe the memory layouts, numerical conventions and
VPU constraints that apply across the library.

- **Standard tensor layout**: row-major, later dimensions fastest, matching C array order. Element ``A[i,j,k]`` is at byte offset ``(i*s1 + j*s2 + k) * element_size``.
- **VPU saturation**: the XS3 VPU clamps results to symmetric bounds rather than rolling over — 8-bit ``[-127, 127]``, 16-bit ``[-32767, 32767]``, 32-bit ``[-2147483647, 2147483647]``. Inner products are therefore not associative. See `VPU saturating arithmetic <https://www.xmos.com/documentation/XM-015059-UG/html/doc/rst/src/reference/notes.html#note-vpu-saturating-arithmetic>`_ and the `XS3 ISA reference <https://www.xmos.com/documentation/XM-014007-PS/html/doc/rst/xs3-arch-inst.html>`_ for full VPU instruction details.
- **Accumulation and output scaling**: convolution accumulates 8-bit products into a 32-bit accumulator seeded with a 32-bit bias, then applies: ``y[i] = ((acc32[i] >> shr1[i]) * scale[i]) >> shr2[i]``, with an additional ``>> 8`` for 8-bit outputs. Shifts are saturating and rounding; negative accumulators never shift to zero.
- **Channel groups**: the VPU processes ``VPU_INT8_EPV = 32`` input channels per load and holds ``VPU_INT8_ACC_PERIOD = 16`` accumulators. Parameter tensors are grouped accordingly: input channel groups of 32, output channel groups of 16.
- **BSO tensor layout**: the Bias-Scale-Offset tensor packs the per-channel output parameters required after accumulation into a single 3-D buffer of shape ``(ceil(C_out/16), 7, 16)``. Axis 0 is the output channel group, axis 2 is the channel offset within that group (so channel ``k`` is at ``[k//16, :, k%16]``), and axis 1 selects the parameter: 0 = bias high half-word, 1 = bias low half-word, 2 = shift1, 3 = scale, 4 = offset scale, 5 = offset, 6 = shift2. The interleaved layout lets the VPU load all parameters for a channel group in one pass.

*************
API Reference
*************

nn_pooling.h
============

.. doxygenfile:: nn_pooling.h
   :project: lib_nn

nn_layers.h
===========

.. doxygenfile:: nn_layers.h
   :project: lib_nn

Where the number of output (input) channels is not a multiple of 16 (32) -- and where the function allows this -- there will be an output (input) channel tail. The tail is the last channels which do not form a complete group. Some tensors, in particular the bias-shift-scale tensors, require that tails be padded. Whether padding must be zeros, or if it is safe to use arbitrary values, is specified by the function.
