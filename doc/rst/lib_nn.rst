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

    #include "nn_pooling.h"
    #include "nn_layers.h"

*******
Example
*******

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

*****************
Library Structure
*****************

Layers
======

Geometry
========

Parameter Preparation
=====================

Aggregation and Output Transformation
=====================================

VPU Support
===========

*************
API Reference
*************

