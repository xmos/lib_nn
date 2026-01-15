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

    Output Tensor:
    101 202 303 404 505 606 707 808 909 1010 1111 1212
