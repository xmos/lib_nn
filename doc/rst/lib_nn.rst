:tocdepth: 3

##############################
lib_nn: Neural network library
##############################

************
Introduction
************

``lib_nn`` is a library of optimised kernels for the neural network operators commonly used in 8-bit quantised inference, such as convolution, pooling, fully-connected layers and elementwise operations. 
Each kernel is written to maximise performance and minimise memory footprint on XMOS devices.

This library targets the xs3 and vx4 architectures. These architectures have a vector unit with 256-bit wide registers that can operate in 8-bit, 16-bit or 32-bit integer mode; ``lib_nn`` kernels are written to make direct use of this vector unit, alongside portable C reference implementations of the same operators.

This document assumes familiarity with the XMOS xCORE architecture, the XMOS tool chain, the C programming language, and neural network concepts. 

*****
Usage
*****

``lib_nn`` is intended to be used with the `XCommon CMake <https://www.xmos.com/file/xcommon-cmake-documentation/?version=latest>`_
, the `XMOS` application build and dependency management system.

To use this library in an application include ``lib_nn`` in the application's ``APP_DEPENDENT_MODULES`` list in `CMakeLists.txt`, for example:

.. code-block:: cmake

    set(APP_DEPENDENT_MODULES "lib_nn")

.. note:: Dependent modules should be pinned to release versions where possible, otherwise the
   latest commit on the `develop` branch will be used.  For further details on managing modules,
   pinning to a release version and other options, please see the page `xcommon-cmake Dependency Management <https://www.xmos.com/documentation/XM-015090-PC/html/doc/dependency_management.html>`_.

``lib_nn`` functions are accessed via their respective header files, for example:

.. code-block:: C

    #include "nn_layers.h"

*******
Example
*******

The ``examples/add_tensor`` directory contains a minimal application that demonstrates how to use ``lib_nn``.

The program prints the two input tensors and the result of adding them together element-wise.

This example is built for ``XK-EVK-XU316`` target and runs it in the xcore simulator.

First, make sure the XMOS XTC tools are installed and activated.

Then, from the top level of the repository, run the following commands::

    cd examples/add_tensor
    cmake -G "Unix Makefiles" -B build 
    xmake -C build

This will build ``add_tensor.xe``. To run this binary in simulation use::

    xsim bin/add_tensor.xe

After running the input tensors and the addition result are printed to the console::

    Input1: -100,   200,    300,    400,    -500,   800,    100,    -50,    -25,    1000,   1100,   1200,
    Input2: 100,    200,    300,    400,    500,    600,    700,    800,    900,    1000,   1100,   1200,
    Output: 0,      400,    600,    800,    0,      1400,   800,    750,    875,    2000,   2200,   2400,

Output correspond to input1 and input2 added together.

*****************
Library Structure
*****************

``lib_nn`` separates operator execution, parameter preparation and hardware
support so callers can reuse prepared data and kernel components without
having to implement VPU operations themselves.

The library is organised around five responsibilities. Layers provide callable
tensor operations. Geometry describes image shapes and filter windows.
Parameter preparation converts scales, biases and weights into kernel-ready
data. Aggregation and output transformation compute intermediate results and
convert them into output values. VPU support provides vector types, memory
utilities and instruction simulation for reference implementations. These are
related components, not stages that every operator must pass through.

For a direct layer call, supply the tensor buffers and any required shape or
prepared parameters. For a composed filter kernel, select the input-access,
aggregation and output-transform functions, prepare their parameters and
weights, then call ``nn::execute()`` for the selected output region. The build
selects the reference or target-specific implementations used by the kernels.

For example, a padded int8 Conv2D follows this flow:

.. code-block:: text

    Setup:
    geometry + weights + scales and biases
        -> prepared kernel parameters and reordered weights
        -> nn::execute() for the selected output region

    Processing inside nn::execute():
    input tensor -> gather padded window -> multiply by weights
                 -> scale, add bias and clip -> output tensor

The three processing stages are ``memcpyfn_imtocol_padded()``,
``mat_mul_generic_int8()`` and ``otfn_int8()``, connected by ``conv_params_t``.
``nn::execute()`` repeats them across output pixels and channel groups,
reusing each gathered window across the groups for that pixel. The complete
setup is demonstrated in ``test/integration/src/test_Conv2dRegression.cpp``.

Layers
======

Layers extract features, combine intermediate results and adapt tensor
representations for inference without requiring application-level kernel
implementations.

``lib_nn`` provides direct tensor operations and composable kernels for
convolution and max pooling. The main operations are described below.

Applications supply tensor buffers, prepare any required parameters or tables,
and execute operations in the order required by the network.

- **Convolution and matrix multiplication** form weighted combinations of
    inputs. Convolution applies kernels to successive tensor windows; matrix
    multiplication computes dot products between matrix rows and columns.

- **Elementwise arithmetic** merges feature paths or applies scaling and
    gating. It adds or multiplies corresponding elements of two tensors,
    accounting for their quantisation scales.

- **Pooling** summarises local windows independently for each channel.
    Max pooling retains the largest value; average pooling computes the mean.
    Neither uses learned weights, and window size and stride determine the
    output dimensions. Max pooling uses dedicated kernels. Average pooling
    reuses depthwise convolution with fixed all-one filters, scaling each
    channel's accumulated sum to produce the mean.

- **Activations** introduce non-linear responses. ReLU, sigmoid and tanh act
    independently on each element. Softmax converts a vector of scores into
    normalised exponentials, with each output depending on the entire vector.
    Elementwise mappings can use prepared tables or polynomial approximations.

- **Reductions** summarise values or select a result. Mean averages over a
    dimension; argmax returns the index of the largest value. Mean computation
    uses a caller-supplied factor for averaging and quantisation rescaling.

- **Quantisation** reduces storage and enables integer VPU computation by
    converting floating-point values to scaled integers. The scale is the real
    step between adjacent integers; smaller scales trade range for precision.
    The int16 conversions use zero point zero: divide by the scale, round and
    clip to the integer range. For example, ``1.0`` at scale ``0.1`` becomes ``10``.

- **Requantisation** connects integer tensors with different scales without
    returning to floating point. It multiplies values by
    ``input_scale / output_scale``, then rounds and clips. Thus ``10`` at scale
    ``0.1`` becomes ``5`` at scale ``0.2``, representing the same real value.

- **Dequantisation** makes integer results available to floating-point
    consumers by multiplying by the scale. Thus ``10`` at scale ``0.1`` becomes
    approximately ``1.0``. It cannot recover precision lost through quantisation.

- **Expansion** adapts tensors to wider integer element types by
    sign-extending values, without changing their integer values or scales.

- **Padding** adapts packed data to a kernel's storage layout by inserting
    specified bytes, for example expanding three-byte blocks to four bytes.
    This is distinct from padding a convolution's input window.

Geometry
========

Geometry separates tensor shapes and coordinate calculations from kernel
arithmetic. It describes which input values contribute to an output and which
output region to compute, allowing work to be divided without changing the
operation itself.

The logical tensor is distinct from its memory representation. Geometry
describes dimensions and coordinates; a kernel receives a buffer and the
parameters that describe its layout. For a row-major image with channels
innermost, element ``X[r,c,p]`` has element offset
``(r * width + c) * channels + p``. The byte offset also depends on the element
size. Reordered weights and packed parameter buffers have kernel-specific
layouts rather than this image layout.

For two-dimensional, channel-based operations, the C++ types in ``api/geom/``
describe the following:

- **ImageGeometry** specifies rows, columns, channels and element size, and
    provides memory-stride calculations.
- **WindowGeometry** specifies the window shape, starting position, stride
    and dilation.
- **Filter2dGeometry** combines the input, output and window descriptions.
- **WindowLocation** maps a particular output coordinate to its input window.
- **ImageRegion** selects a rectangular range of rows, columns and channels.

For example, split an operation's output between two kernel invocations.
The output geometry describes the complete tensor; each region selects the
part that one invocation should compute:

.. code-block:: cpp

    #include "geom/ImageGeometry.hpp"

    const nn::ImageGeometry output(8, 16, 16);
    const int split_row = output.height / 2;

    const nn::ImageRegion first_half(
        0, 0, 0, split_row, output.width, output.depth);
    const nn::ImageRegion second_half(
        split_row, 0, 0, output.height - split_row, output.width, output.depth);

``output`` has 8 rows, 16 columns and 16 channels. Region arguments specify
the starting row, column and channel, followed by the counts in each dimension.
``first_half`` selects rows ``[0, 4)`` and ``second_half`` selects rows
``[4, 8)``; both include every column and channel. Together they cover the
output exactly once.

During kernel setup, use each region to prepare a separate invocation's output
bounds, while keeping the same full tensor geometry and operation parameters.
The invocations can run sequentially or on separate cores. Parallel invocations
must use separate scratch buffers where required. The regions describe work;
they neither allocate buffers nor launch execution. Region boundaries must
satisfy the selected kernel's alignment and channel-group requirements.

Parameter Preparation
=====================

Parameter preparation separates setup calculations from tensor processing.
Preparation functions convert scales, zero points, biases and weights into
the representations required by a selected kernel. Execution functions consume
this prepared data, avoiding repeated conversion inside the processing loop.
The data can be reused while the operation parameters and target remain
unchanged.

Preparation takes several forms:

- **Parameter blobs** encode transformed scalar parameters. For example,
    ``quantize_int16_tensor_blob()`` converts an output scale into the blob
    consumed by ``quantize_int16_tensor()``. The API recommends generating
    these blobs at build time for use at run time; they are generated by
    preparation functions, not implicitly by the compiler. Blob sizes,
    alignment requirements and preparation failure conditions are specified
    in ``api/nn_layers.h``.
- **Weight reordering** arranges convolution weights in the order consumed by
    the aggregation kernel. ``MatMulInt8::reorder_kernel_weights()`` prepares
    weights for the generic int8 path; other aggregation variants have their
    own layout requirements.
- **Output parameters** represent scaling and bias in fixed-point form and
    pack per-channel values for the chosen output transform. The preparation
    helpers in ``api/OutputTransformFn.hpp`` support the corresponding int8
    and binary transforms.

Grouping and packing reflect how the VPU processes data. A 256-bit load holds
32 int8 elements, while int8 multiply-accumulate operations maintain a group
of 16 accumulators. Loaded elements are not necessarily distinct channels:
their meaning depends on the input and weight layout. Prepared weights and
output parameters follow the selected kernel's grouping and padding rules;
formats for different aggregation or output-transform variants are not
interchangeable.

Aggregation and Output Transformation
=====================================

Separating the calculation from output conversion allows an aggregation kernel
to be reused with different quantisation schemes and output representations.
For example, the same int8 convolution accumulator can feed either a shared-
shift or a channelwise-shift output transform.

The two stages have distinct responsibilities:

- **Aggregation** combines the input window into intermediate results for an
    output channel group. Convolution computes sums of products; max pooling
    computes channel maxima. Results go into ``VPURingBuffer``, not directly
    into the output tensor. Convolution stores accumulator halves there;
    max pooling stores its int8 maxima in the buffer's ``vR`` vector.
- **Output transformation** reads those results and writes the output tensor.
    Depending on the operation, it applies prepared scaling and bias, rounds
    and saturates, produces thresholded bits, or simply stores pooling results.

Select compatible input-access, aggregation and output-transform functions,
and connect them with their parameter structs in ``conv_params_t``.
``nn::execute()`` calls the input handler, then aggregation and output
transformation for each output channel group. Dense convolution reuses the
input patch across groups; depthwise execution selects the input channels
for each group. Shared buffer storage does not make all variants interchangeable:
their element types, weight layouts and output parameters must agree.

**Aggregation variants** describe how the input is accessed and combined:

- ``mat_mul_generic_*()`` consumes a contiguous input patch, usually gathered
    into scratch memory. ``mat_mul_direct_*()`` traverses the original tensor
    using prepared strides, avoiding that copy when its constraints permit.
- ``mat_mul_dw_direct()`` and its int16 variant compute depthwise products:
    each output channel uses its corresponding input channel, rather than
    combining all input channels.
- Binary variants compute inner products of packed one-bit inputs and weights.
    Integer variants cover int8, int16 and mixed int16-input/int8-weight paths;
    the mixed path is not implemented on XS3.
- ``maxpool_direct()`` computes maxima without weights rather than a sum of
    products.

**Output variants** determine what is stored:

- ``otfn_int8()`` uses shared initial and final shifts with per-channel
    multipliers and biases. ``otfn_int8_channelwise()`` adds a separate initial
    shift per channel to accommodate different accumulator ranges.
- ``output_transform_fn_int16()`` adds prepared accumulator-domain biases,
    applies fixed-point multipliers and saturates the results to int16.
- ``otfn_binary()`` thresholds results and packs one bit per channel.
    ``otfn_int8_clamped()`` provides the offset, clamp and scaling path used
    to produce int8 outputs from binary aggregation.
- ``otfn_int8_maxpool()`` stores the selected maxima without rescaling them.

The aggregation API is in ``api/AggregateFn.hpp``; its implementations are in
``src/cpp/AggregateFn.cpp``, ``src/cpp/AggregateFn_DW.cpp`` and
``src/cpp/MaxPool.cpp``. Output-transform helpers and int8/binary entry points
are in ``api/OutputTransformFn.hpp`` and ``src/cpp/OutputTransformFn.cpp``.
The int16 output API is in ``api/nn_layers.h``, with implementation in
``src/c/output_transform_fn_int16.c``. Corresponding files in ``src/asm/``
provide target-specific implementations of these stages; the wrappers select
them instead of the reference path when ``NN_USE_REF`` is not enabled.

VPU Support
===========

The VPU (Vector Processing Unit) provides specialized instructions for
efficient vector arithmetic. The ``vpu_sim`` module provides a software model
of these instructions. The library also provides vectorized memory operations
that mirror common ``<string.h>`` routines, such as ``memcpy``, ``memmove`` and
``memset``. These helpers have specific constraints; see each function's
documentation for details.

VPU Simulation
--------------

Reference kernels need to reproduce vector arithmetic without requiring VPU
hardware. Simulation allows these kernels to run on a host for testing and
debugging; it models instruction results, not execution timing.

Reference implementations use an explicit ``vpu_t`` state and instruction-like
functions: ``VSETC()`` selects the element mode, loads populate registers,
multiply-accumulate and shift operations update them, and stores copy results
back to memory. ``NN_USE_REF`` selects reference kernel paths. Applications
using those kernels do not need to manage simulated registers themselves.

``api/vpu_sim.h`` defines the register state and operations;
``src/c/vpu_sim.c`` implements vector loads, arithmetic, accumulator rotation,
rounding, saturation and output conversion. The ``nn::VPU`` class wraps the
same operations for C++, and register-printing helpers support debugging.

VPU Memory Operations
---------------------

Moving weights and tensors or initialising scratch buffers can be a significant
part of kernel execution. The memory helpers provide vector-based paths for
these transfers and fills without implementing copy loops in each kernel.

The library mirros common memory operations such as:

- **Copy:** ``vpu_memcpy_int()`` is tuned for internal SRAM and ``vpu_memcpy_ext()`` for external memory. 
- **Move:** ``vpu_memmove_word_aligned()`` takes a byte count and permits overlapping source and destination regions.
- **Fill:** ``vpu_memset_32()`` and ``vpu_memset_vector()`` both repeat a
    32-bit value, with the fill length specified in words or 32-byte vectors,
    respectively. ``vpu_memset_256()`` fills a specified number of bytes from
    a 32-byte pattern buffer; ``broadcast_32_to_256()`` prepares such a buffer
    by repeating a 32-bit value across it.

Notes
=====

- **Word alignment:** many XS3 kernels require tensor and parameter buffers
    to be word aligned, meaning their addresses are multiples of four bytes.
    For example, ``quantize_int16_tensor()`` requires aligned input, output
    and blob buffers on XS3. Requirements are specified per function; memory
    helpers may impose alignment constraints on both architectures.

- **VPU width:** both architectures have 256-bit vector registers, holding
    32 int8, 16 int16 or 8 int32 elements. Vector width is distinct from the
    number of accumulators and does not imply that each load spans that many
    tensor channels.

- **Saturation ranges:** XS3 symmetric saturation uses ``[-127, 127]`` for
    int8 and ``[-32767, 32767]`` for int16. VX4 output depth conversions use
    the full two's-complement ranges, ``[-128, 127]`` and ``[-32768, 32767]``,
    respectively. Bounds depend on the instruction and kernel path; these
    ranges are not a universal rule for every arithmetic operation.

- **Accumulation order:** saturation clips intermediate results rather than
    wrapping them. If an intermediate sum saturates, a different accumulation
    order can produce a different final result.

- **Output conversion:** bias, fixed-point scaling, rounding and clipping
    are applied according to the selected output transform. Shared-shift,
    channelwise-shift and int16 transforms have different parameter formats
    and numerical behaviour.

- **Zero points:** the int16 quantisation and dequantisation functions use
    zero point zero, with the relationship ``x = q * s`` between the represented
    real value, integer and scale. Other operators may account for nonzero
    zero points through prepared biases and output parameters.

- **Reference behaviour:** ``NN_USE_REF`` selects reference implementations,
    whose simulation functions model instruction-specific rounding and
    saturation. Some output conversions select full-range saturation when
    ``NN_USE_REF`` is enabled, so a host reference build does not by itself
    establish bit-identical behaviour at the XS3 negative limit.

*************
API Reference
*************

Layer Operations
================

Tensor operations and their parameter-preparation functions.

.. doxygenfile:: nn_layers.h
   :project: lib_nn

Tensor and Window Types
=======================

C types describing tensors, images, convolution windows, output jobs and
binary data.

.. doxygenfile:: nn_types.h
    :project: lib_nn

.. doxygenfile:: nn_image.h
    :project: lib_nn

.. doxygenfile:: nn_conv2d_structs.h
    :project: lib_nn

.. doxygenfile:: nn_window_params.h
    :project: lib_nn

.. doxygenfile:: nn_bin_types.h
    :project: lib_nn

Geometry
========

C++ descriptions of image and window shapes, coordinate mappings, output
regions and padding.

.. doxygenfile:: ImageGeometry.hpp
    :project: lib_nn

.. doxygenfile:: WindowGeometry.hpp
    :project: lib_nn

.. doxygenfile:: Filter2dGeometry.hpp
    :project: lib_nn

.. doxygenfile:: WindowLocation.hpp
    :project: lib_nn

.. doxygenfile:: geom/util.hpp
    :project: lib_nn

Kernel Composition
==================

Execution interfaces, input-access handlers, aggregation kernels, output
transforms and the accumulator buffer shared between stages.

.. doxygenfile:: AbstractKernel.hpp
    :project: lib_nn

.. doxygenfile:: MemCpyFn.hpp
    :project: lib_nn

.. doxygenfile:: AggregateFn.hpp
    :project: lib_nn

.. doxygenfile:: OutputTransformFn.hpp
    :project: lib_nn

.. doxygenfile:: vpu.hpp
    :project: lib_nn

Convolution Preparation
=======================

TFLite parameter conversion and transpose-convolution weight preparation.

.. doxygenfile:: conv2d_utils.hpp
    :project: lib_nn

.. doxygenfile:: TransposeConv.hpp
    :project: lib_nn

Work Partitioning
=================

Utilities for splitting aligned ranges between threads.

.. doxygenfile:: nn_op_utils.h
    :project: lib_nn

VPU Memory Operations
=====================

Vector-based copy, overlapping move and fill operations.

.. doxygenfile:: vpu_mem.h
    :project: lib_nn

VPU Simulation Support
======================

Instruction simulation, register state and VPU constants.

.. doxygenfile:: xs3_vpu.h
    :project: lib_nn

.. doxygenfile:: vpu_sim.h
    :project: lib_nn

Configuration
=============

Target selection and saturation configuration.

.. doxygenfile:: nn_arch.h
    :project: lib_nn

.. doxygenfile:: nn_config.h
    :project: lib_nn
