Add Tensor Example
==================

This is a basic example that adds two tensors together.
The purpose of this example is to demonstrate how to use the `lib_nn` library to perform tensor addition.

First, make sure XMOS tools are properly installed and activated on your system.

To build and run the example, follow these steps from the current directory:

.. code-block:: console
    
    # build 
    cmake -G "Unix Makefiles" -B build
    cmake --build build
    # run
    xsim bin/add_tensor.xe

The program will output the result of adding the two input tensors element-wise. 
The expected output is:

.. code-block:: console
    
    Input1: -100,   200,    300,    400,    -500,   800,    100,    -50,    -25,    1000,   1100,   1200,
    Input2: 100,    200,    300,    400,    500,    600,    700,    800,    900,    1000,   1100,   1200,
    Output: 0,      400,    600,    800,    0,      1400,   800,    750,    875,    2000,   2200,   2400,
